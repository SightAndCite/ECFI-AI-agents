"""Testbed invariants.

These are the properties the experiments silently assume.  Each one, if it
broke, would produce a plausible-looking but meaningless result, so they are
asserted rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecfi.detect import (
    alternation_amplitude, lag1_autocorr, margin_knee, nyquist_fraction,
    variance_extrapolation,
)
from ecfi.identify import build_design, identify
from ecfi.policies import CongestionAwarePolicy, ProportionalPolicy, make_policy
from ecfi.schedules import Schedule, constant, dither, pulse_train
from ecfi.sim import RESOURCE_COUPLED, WAIT, SimConfig, run_sim


def small(**kw) -> SimConfig:
    base = dict(W=100, n_windows=90, warmup_windows=20, seed=1)
    base.update(kw)
    return SimConfig(**base)


# --- determinism and shape -------------------------------------------------


def test_runs_are_reproducible_given_a_seed():
    a = run_sim(small(kappa=3.0))
    b = run_sim(small(kappa=3.0))
    assert np.allclose(a.u, b.u) and np.allclose(a.v, b.v) and np.allclose(a.q, b.q)


def test_different_seeds_differ():
    a = run_sim(small(kappa=3.0, seed=1))
    b = run_sim(small(kappa=3.0, seed=2))
    assert not np.allclose(a.u, b.u)


def test_window_count_and_ranges():
    cfg = small(kappa=2.0)
    ws = run_sim(cfg)
    assert len(ws) == cfg.n_windows - cfg.warmup_windows
    assert np.all((ws.u >= 0) & (ws.u <= 1))
    assert np.all(ws.v >= 0) and np.all(ws.v <= cfg.v_max)
    assert np.all(ws.q >= 0)
    assert np.all(np.diff(ws.t_end) > 0)


# --- the couplings the paper depends on ------------------------------------


def test_wait_mode_never_falls_back():
    ws = run_sim(small(kappa=3.0, mode=WAIT, bg_load=constant(3.0)))
    assert ws.fallback.sum() == 0


def test_resource_coupling_produces_fallbacks_under_load():
    ws = run_sim(small(kappa=3.0, mode=RESOURCE_COUPLED, bg_load=constant(4.0)))
    assert ws.fallback.mean() > 0


def test_background_load_raises_latency():
    quiet = run_sim(small(kappa=2.0, mode=WAIT))
    busy = run_sim(small(kappa=2.0, mode=WAIT, bg_load=constant(3.0)))
    assert busy.wait.mean() > quiet.wait.mean()


def test_store_switch_is_the_negative_control():
    """Disabling the store must remove the epistemic propagation path."""
    on = run_sim(small(kappa=4.0, store_enabled=True))
    off = run_sim(small(kappa=4.0, store_enabled=False))
    # with no contamination the error rate is lower and less autocorrelated
    assert off.u.mean() < on.u.mean()


def test_zero_gain_holds_effort_constant():
    ws = run_sim(small(kappa=0.0))
    assert ws.v.std() == pytest.approx(0.0, abs=1e-12)


def test_gain_increases_effort_variability():
    lo = run_sim(small(kappa=1.0))
    hi = run_sim(small(kappa=6.0))
    assert hi.v.std() > lo.v.std()


def test_more_verification_reduces_error_at_fixed_nominal_effort():
    """Raising nominal effort must lower the error rate -- c > 0 by construction."""
    weak = run_sim(small(kappa=0.0, v_bar=0.6, v_half=1.3))
    strong = run_sim(small(kappa=0.0, v_bar=2.2, v_half=1.3))
    assert strong.u.mean() < weak.u.mean()


# --- schedules -------------------------------------------------------------


def test_schedule_is_right_continuous_piecewise_constant():
    s = Schedule([(0.0, 1.0), (10.0, 5.0)])
    assert s.at(0.0) == 1.0 and s.at(9.99) == 1.0
    assert s.at(10.0) == 5.0 and s.at(1e6) == 5.0


def test_pulse_train_alternates():
    s = pulse_train(base=0.0, high=2.0, period=10.0, duty=0.5, horizon=30.0)
    assert s.at(1.0) == 2.0 and s.at(7.0) == 0.0 and s.at(11.0) == 2.0


def test_dither_stays_above_floor():
    rng = np.random.default_rng(0)
    s = dither(base=1.0, amplitude=5.0, block=1.0, horizon=50.0, rng=rng, lo=0.25)
    assert all(v >= 0.25 for _, v in s.breakpoints)


# --- policies --------------------------------------------------------------


def test_policies_saturate():
    p = ProportionalPolicy(kappa=100.0, v_bar=1.0, u_ref=0.0, v_max=4.0)
    assert p.effort(1.0) == 4.0
    assert p.effort(-1.0) == 0.0


def test_congestion_aware_reduces_to_proportional_at_zero_gamma():
    kw = dict(kappa=3.0, v_bar=1.3, u_ref=0.12, v_max=4.0)
    a = make_policy(gamma=0.0, **kw)
    b = CongestionAwarePolicy(gamma=0.0, **kw)
    assert isinstance(a, ProportionalPolicy)
    assert a.effort(0.3, 5.0) == pytest.approx(b.effort(0.3, 5.0))


def test_congestion_aware_damps_with_queue():
    p = CongestionAwarePolicy(kappa=5.0, gamma=10.0, v_bar=1.0, u_ref=0.0, v_max=9.0)
    assert p.effort(0.3, q_obs=2.0) < p.effort(0.3, q_obs=0.0)


# --- identification --------------------------------------------------------


def test_lead_alignment_changes_the_design_matrix():
    ws = run_sim(small(kappa=3.0))
    Xu_lead, *_ = build_design([ws], lead_inputs=True)
    Xu_raw, *_ = build_design([ws], lead_inputs=False)
    assert Xu_lead.shape == Xu_raw.shape
    assert not np.allclose(Xu_lead, Xu_raw)


def test_identification_recovers_positive_couplings():
    """With proper excitation every coupling must come out with the right sign."""
    ref = SimConfig(W=200)
    wt = ref.W / ref.lam()
    horizon = 260 * wt
    series = []
    for seed in (1, 2):
        rng = np.random.default_rng(seed)
        series.append(run_sim(SimConfig(
            kappa=3.0, W=200, n_windows=220, warmup_windows=40, seed=seed,
            bg_load=pulse_train(0.0, 3.0, 17 * wt, 0.4, horizon, 3 * wt),
            ambiguity=pulse_train(0.0, 0.8, 23 * wt, 0.3, horizon, 11 * wt),
            v_bar_sched=dither(1.3, 0.6, 5 * wt, horizon, rng, lo=0.2),
            amb_true_boost=0.25, amb_u_boost=0.25,
        )))
    res = identify(series)
    p = res.params
    assert p.b > 0 and p.c > 0 and p.e > 0 and p.f > 0
    assert 0.0 <= p.a < 1.0 and 0.0 <= p.d < 1.0
    assert res.kappa_crit > 0
    assert res.reg_u.r2 > 0.1 and res.reg_q.r2 > 0.1


# --- detectors -------------------------------------------------------------


def test_nyquist_fraction_detects_pure_alternation():
    x = np.array([1.0, -1.0] * 64)
    assert nyquist_fraction(x) > 0.99
    rng = np.random.default_rng(0)
    assert nyquist_fraction(rng.normal(size=128)) < 0.2


def test_lag1_autocorr_signs():
    assert lag1_autocorr(np.array([1.0, -1.0] * 64)) == pytest.approx(-1.0, abs=0.02)
    assert lag1_autocorr(np.cumsum(np.ones(128))) > 0.9


def test_alternation_amplitude_recovers_the_amplitude():
    amp = 0.4
    x = amp * np.array([1.0, -1.0] * 64)
    assert alternation_amplitude(x) == pytest.approx(2 * amp, rel=0.02)


def test_margin_knee_finds_the_flattening_point():
    ks = np.linspace(0, 10, 21)
    m = 1.0 * np.exp(-ks) + 0.05
    knee = margin_knee(ks, m)
    assert 2.0 < knee < 6.0


def test_variance_extrapolation_recovers_a_linear_inverse():
    ks = np.linspace(0, 4, 9)
    inv = 1.0 - 0.2 * ks           # hits zero at kappa = 5
    assert variance_extrapolation(ks, 1.0 / inv) == pytest.approx(5.0, rel=0.05)
