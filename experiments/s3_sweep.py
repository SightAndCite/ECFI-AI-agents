"""S3 -- out-of-sample stability-boundary test (Section V-3, VI-S3).

Reads the *frozen* estimates from ``frozen/kcrit.json`` and sweeps the
verification gain over ``[0, 1.5 kappa_hat_crit]`` at several task-arrival
intensities.  The pre-registered endpoint is the onset of sustained period-two
alternation or overload, detected as a spectral peak at the Nyquist frequency
of ``v_k`` (or divergence of the windowed variance).

Two safeguards keep the test honest:

*  the detection threshold is calibrated on the **negative control** (shared
   context store disabled, so ``b ~ 0`` and no flip is possible by
   construction), never on the arm being judged;
*  ``kappa_hat_crit`` is read from disk and never recomputed here.

The primary result is the *distance* between the observed transition and the
frozen prediction.  A transition far from the prediction is reportable evidence
against the framework (Section VI, outcome ii) and must be published as such.

Usage:  python experiments/s3_sweep.py [--jobs N] [--seeds N] [--grid N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.detect import (  # noqa: E402
    alternation_amplitude, estimate_transition, lag1_autocorr, margin_knee,
    nyquist_fraction, online_margin, variance_extrapolation,
)
from ecfi.runner import FROZEN, RESULTS, default_jobs, load_json, run_many, save_json  # noqa: E402
from ecfi.sim import SimConfig  # noqa: E402
from regimes import get_regime  # noqa: E402

LOAD_FACTORS = (0.45, 0.55, 0.65)
CONTROL_QUANTILE = 0.95      # threshold = this quantile of the negative control


def metrics(ws, baseline_var_u: float) -> dict:
    m = online_margin(ws.u, ws.q, ws.v)
    return dict(
        u=float(ws.u.mean()),
        v=float(ws.v.mean()),
        q=float(ws.q.mean()),
        fallback=float(ws.fallback.mean()),
        wait_p95=float(ws.wait_p95.mean()),
        tokens=float(ws.tokens.mean()),
        sd_v=float(ws.v.std()),
        sd_u=float(ws.u.std()),
        nyq_v=nyquist_fraction(ws.v),
        nyq_u=nyquist_fraction(ws.u),
        acf1_u=lag1_autocorr(ws.u),
        acf1_v=lag1_autocorr(ws.v),
        amp_v=alternation_amplitude(ws.v),
        # ratio is taken on u, not v: at kappa = 0 the effort is constant by
        # construction so var(v) = 0 and a v-ratio would be meaningless
        var_ratio_u=float(np.var(ws.u)) / baseline_var_u if baseline_var_u > 0 else float("nan"),
        margin=float(np.nanmean(m)) if np.any(np.isfinite(m)) else float("nan"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--grid", type=int, default=13)
    ap.add_argument("--control-seeds", type=int, default=8)
    ap.add_argument("--windows", type=int, default=320)
    ap.add_argument("--regime", default="shallow")
    args = ap.parse_args()

    regime = get_regime(args.regime)
    ov = regime.overrides()
    kc_path = FROZEN / regime.name / "kcrit.json"
    if not kc_path.exists():
        print(f"{kc_path} is missing. Run experiments/s2_identify.py "
              f"--regime {regime.name} first:\n"
              "the whole point of S3 is that the prediction was fixed beforehand.")
        return 1
    frozen = load_json(kc_path)
    k_hat = float(frozen["kappa_crit"])
    if not np.isfinite(k_hat) or k_hat <= 0:
        print(f"frozen kappa_crit is not usable ({k_hat}); "
              "report this as a failed identification, not as a sweep result.")
        return 1

    kappas = list(np.linspace(0.0, 1.5 * k_hat, args.grid))
    print(f"frozen kappa_hat_crit = {k_hat:.4f}  (frozen at {frozen.get('frozen_at')})")
    print(f"sweeping kappa over [0, {1.5 * k_hat:.3f}] at {args.grid} points, "
          f"{args.seeds} seeds, load factors {LOAD_FACTORS}")

    def cfg(kappa, load, seed, store=True):
        return SimConfig(**ov, kappa=kappa, load_factor=load, store_enabled=store,
                         n_windows=args.windows, warmup_windows=60, seed=seed)

    # --- negative control: store disabled, so no flip is possible ----------
    ctrl_cfgs = [cfg(k, 0.55, s, store=False)
                 for k in kappas for s in range(1, args.control_seeds + 1)]
    ctrl = run_many(ctrl_cfgs, jobs=args.jobs, label="control")
    ctrl_nyq = [nyquist_fraction(ws.v) for ws in ctrl]
    threshold = float(np.quantile(ctrl_nyq, CONTROL_QUANTILE))
    print(f"\nnegative-control Nyquist fraction: median {np.median(ctrl_nyq):.4f}, "
          f"q{int(CONTROL_QUANTILE*100)} {threshold:.4f}  -> detection threshold")

    payload = {
        "regime": regime.name, "regime_overrides": ov,
        "frozen": frozen, "kappa_hat_crit": k_hat, "kappas": kappas,
        "load_factors": list(LOAD_FACTORS), "seeds": args.seeds,
        "threshold": threshold, "control_quantile": CONTROL_QUANTILE,
        "control_nyq": ctrl_nyq, "by_load": {},
    }

    for load in LOAD_FACTORS:
        base_cfgs = [cfg(0.0, load, s) for s in range(1, args.seeds + 1)]
        base = run_many(base_cfgs, jobs=args.jobs, label=f"baseline lf={load}")
        baseline_var_u = float(np.mean([np.var(ws.u) for ws in base])) or 1e-12

        cfgs = [cfg(k, load, s) for k in kappas for s in range(1, args.seeds + 1)]
        series = run_many(cfgs, jobs=args.jobs, label=f"sweep lf={load}")

        per_kappa, per_seed_nyq = {}, {}
        it = iter(series)
        for k in kappas:
            rows = [metrics(next(it), baseline_var_u) for _ in range(args.seeds)]
            per_seed_nyq[k] = [r["nyq_v"] for r in rows]
            per_kappa[k] = {
                key: [float(np.mean([r[key] for r in rows])),
                      float(np.std([r[key] for r in rows], ddof=1) / np.sqrt(len(rows)))]
                for key in rows[0]
            }

        est = estimate_transition(kappas, per_seed_nyq, threshold)

        # Primary, threshold-free endpoints.  The spectral endpoint above is
        # retained as a secondary readout only: the store-disabled control
        # leaves the corrective feedback loop intact, so it cannot calibrate a
        # spectral threshold for the flip (it controls b, not the loop).
        margins = [per_kappa[k]["margin"][0] for k in kappas]
        varrat = [per_kappa[k]["var_ratio_u"][0] for k in kappas]
        k_knee = margin_knee(kappas, margins)
        k_var = variance_extrapolation(kappas, varrat)

        payload["by_load"][str(load)] = {
            "per_kappa": {str(k): v for k, v in per_kappa.items()},
            "kappa_star_margin_knee": k_knee,
            "kappa_star_var_extrapolation": k_var,
            "ratio_margin_knee": k_knee / k_hat if np.isfinite(k_knee) else None,
            "ratio_var_extrapolation": k_var / k_hat if np.isfinite(k_var) else None,
            "kappa_star_spectral": est.kappa_star,
            "spectral_threshold": threshold,
            "fraction_positive": {str(k): v for k, v in est.fraction_positive.items()},
        }

        print(f"\n== load_factor = {load}")
        print("   kappa    error     v      q    fb    nyq(v)  acf1(u)  varratio  margin")
        for k in kappas:
            m = per_kappa[k]
            print(f"  {k:6.2f}  {m['u'][0]:.4f}  {m['v'][0]:.3f}  {m['q'][0]:.3f}  "
                  f"{m['fallback'][0]:.3f}  {m['nyq_v'][0]:.4f}  {m['acf1_u'][0]:+.3f}   "
                  f"{m['var_ratio_u'][0]:7.2f}  {m['margin'][0]:+.3f}")
        print(f"   predicted kappa_hat_crit      = {k_hat:.3f}")
        print(f"   observed  margin knee         = {k_knee:.3f}"
              f"   ratio {k_knee / k_hat:.2f}" if np.isfinite(k_knee) else
              "   observed  margin knee         = n/a")
        print(f"   observed  variance extrapol.  = {k_var:.3f}"
              f"   ratio {k_var / k_hat:.2f}" if np.isfinite(k_var) else
              "   observed  variance extrapol.  = n/a")
        print(f"   secondary spectral endpoint   : {est.summary()}")

    save_json(RESULTS / f"s3_sweep_{regime.name}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
