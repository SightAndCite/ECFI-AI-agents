"""Transition detection and early-warning signals (Sections III and VI).

The pre-registered endpoint of the out-of-sample sweep is the *onset of
sustained period-two alternation or overload*, operationalised as

*  a spectral peak at the Nyquist frequency of ``v_k`` or ``q_k``
   (:func:`nyquist_fraction`), or
*  divergence of the windowed variance (:func:`variance_ratio`).

The H4 precursors logged on approach are windowed variance, lag-one
autocorrelation, recovery time after a standardised load pulse, and the online
stability margin ``M(t) = 1 - rho(A_hat(t))`` from a recursively identified
model.

Detection thresholds are calibrated on the *negative control* (the store
disabled, where no flip can occur by construction), so that "we saw a
transition" cannot be an artefact of the detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "nyquist_fraction",
    "lag1_autocorr",
    "variance_ratio",
    "alternation_amplitude",
    "recovery_time",
    "online_margin",
    "TransitionEstimate",
    "estimate_transition",
]


def _dm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x - x.mean()


def nyquist_fraction(x: Sequence[float]) -> float:
    """Share of the AC power sitting at the Nyquist frequency.

    For white noise of length N this has expectation ~ 1/(N/2); a period-two
    limit cycle drives it toward 1.  Compared against the negative control
    rather than against an absolute number.
    """
    d = _dm(np.asarray(x, dtype=float))
    if d.size < 8 or not np.any(d):
        return 0.0
    P = np.abs(np.fft.rfft(d)) ** 2
    ac = P[1:]
    total = float(ac.sum())
    return float(ac[-1] / total) if total > 0 else 0.0


def lag1_autocorr(x: Sequence[float]) -> float:
    d = _dm(np.asarray(x, dtype=float))
    denom = float(d @ d)
    if denom <= 0 or d.size < 3:
        return 0.0
    return float((d[:-1] @ d[1:]) / denom)


def variance_ratio(x: Sequence[float], baseline_var: float) -> float:
    """Windowed variance relative to the baseline (kappa = 0) variance."""
    v = float(np.var(np.asarray(x, dtype=float)))
    return v / baseline_var if baseline_var > 0 else float("inf")


def alternation_amplitude(x: Sequence[float]) -> float:
    """Amplitude of the period-two component, in the units of ``x``.

    The Nyquist Fourier coefficient of a pure alternation of peak-to-peak
    amplitude A has magnitude N*A/2, so this returns A directly.
    """
    d = _dm(np.asarray(x, dtype=float))
    if d.size < 8:
        return 0.0
    c = np.fft.rfft(d)[-1]
    return float(2.0 * np.abs(c) / d.size)


def recovery_time(
    x: Sequence[float], pulse_idx: int, baseline: Optional[float] = None,
    tol_frac: float = 0.25, max_lag: int = 40,
) -> float:
    """Windows until ``x`` returns within ``tol_frac`` of its pre-pulse level.

    Critical slowing down near the boundary shows up here before it shows up
    in the mean.  Returns ``inf`` when recovery does not occur within
    ``max_lag`` windows.
    """
    x = np.asarray(x, dtype=float)
    if pulse_idx <= 1 or pulse_idx >= len(x) - 2:
        return float("nan")
    pre = float(np.mean(x[max(0, pulse_idx - 10):pulse_idx])) if baseline is None else baseline
    peak = float(np.max(np.abs(x[pulse_idx:pulse_idx + 5] - pre)))
    if peak <= 0:
        return 0.0
    tol = tol_frac * peak
    for j in range(pulse_idx, min(len(x), pulse_idx + max_lag)):
        if abs(x[j] - pre) <= tol:
            return float(j - pulse_idx)
    return float("inf")


def online_margin(
    u: Sequence[float], q: Sequence[float], v: Sequence[float],
    window: int = 40, ridge: float = 1e-6,
) -> np.ndarray:
    """Sliding-window estimate of ``M(t) = 1 - rho(A_hat(t))``.

    ``A_hat`` is the *closed-loop* two-by-two map identified from (u, q) over
    the trailing window, so no knowledge of the policy is needed -- which is
    what makes it deployable as a monitor.  Its reliability depends on
    persistent excitation; windows in which the regressor is near-singular
    return NaN rather than a confident wrong number.
    """
    u = np.asarray(u, float); q = np.asarray(q, float)
    n = len(u)
    out = np.full(n, np.nan)
    for t in range(window, n):
        uu = u[t - window:t] - u[t - window:t].mean()
        qq = q[t - window:t] - q[t - window:t].mean()
        X = np.column_stack([uu[:-1], qq[:-1]])
        if np.linalg.cond(X) > 1e6:
            continue
        G = X.T @ X + ridge * np.eye(2)
        row_u = np.linalg.solve(G, X.T @ uu[1:])
        row_q = np.linalg.solve(G, X.T @ qq[1:])
        A = np.array([row_u, row_q])
        out[t] = 1.0 - float(np.max(np.abs(np.linalg.eigvals(A))))
    return out


# --------------------------------------------------------------------------- #
# the pre-registered endpoint
# --------------------------------------------------------------------------- #


@dataclass
class TransitionEstimate:
    kappa_star: float
    threshold: float
    metric: str
    per_kappa: Dict[float, float]
    fraction_positive: Dict[float, float]

    def summary(self) -> str:
        return (
            f"transition at kappa* = {self.kappa_star:.3f} "
            f"({self.metric} > {self.threshold:.4f} in a majority of seeds)"
        )


def estimate_transition(
    kappas: Sequence[float],
    per_seed_metric: Dict[float, Sequence[float]],
    threshold: float,
    metric: str = "nyquist_fraction",
    majority: float = 0.5,
) -> TransitionEstimate:
    """First gain at which the endpoint fires in more than ``majority`` of seeds.

    The threshold must be supplied by the caller and must have been calibrated
    on the negative control *before* the sweep -- this function deliberately
    does not derive one from the data it is judging, and it interpolates
    linearly between the last quiet gain and the first firing one rather than
    snapping to the grid.
    """
    ks = sorted(kappas)
    frac = {
        k: float(np.mean([m > threshold for m in per_seed_metric[k]])) for k in ks
    }
    means = {k: float(np.mean(per_seed_metric[k])) for k in ks}

    kstar = float("nan")
    prev = None
    for k in ks:
        if frac[k] > majority:
            if prev is None:
                kstar = k
            else:
                f0, f1 = means[prev], means[k]
                if f1 > f0:
                    w = (threshold - f0) / (f1 - f0)
                    kstar = prev + max(0.0, min(1.0, w)) * (k - prev)
                else:
                    kstar = k
            break
        prev = k

    return TransitionEstimate(kstar, threshold, metric, means, frac)


# --------------------------------------------------------------------------- #
# threshold-free endpoints
# --------------------------------------------------------------------------- #


def margin_knee(
    kappas: Sequence[float], margins: Sequence[float], frac: float = 0.05,
) -> float:
    """Gain at which the online stability margin has essentially bottomed out.

    Theory predicts the transition where ``rho = 1``, i.e. ``M = 0``.  A
    *stochastic* nonlinear system never reaches it: saturation caps the critical
    mode short of -1, so ``M`` descends and then flattens at a positive floor.
    The measurable analogue of "the eigenvalue has arrived" is therefore the
    knee -- the smallest gain at which ``M`` has come within ``frac`` of its
    total descent:

        M(kappa*) <= M_min + frac * (M(0) - M_min)

    linearly interpolated between grid points.  This needs no control arm and no
    absolute threshold, which matters because the store-disabled negative
    control does *not* remove the corrective feedback loop and so cannot
    calibrate a spectral detector for the flip -- it only removes the epistemic
    propagation path that ``b`` measures.
    """
    ks = np.asarray(kappas, dtype=float)
    ms = np.asarray(margins, dtype=float)
    ok = np.isfinite(ms)
    ks, ms = ks[ok], ms[ok]
    if ks.size < 3:
        return float("nan")
    order = np.argsort(ks)
    ks, ms = ks[order], ms[order]
    m0, mmin = float(ms[0]), float(np.min(ms))
    if m0 <= mmin:
        return float("nan")
    level = mmin + frac * (m0 - mmin)
    for i in range(1, len(ks)):
        if ms[i] <= level:
            m_prev, m_cur = ms[i - 1], ms[i]
            if m_prev == m_cur:
                return float(ks[i])
            w = (m_prev - level) / (m_prev - m_cur)
            return float(ks[i - 1] + max(0.0, min(1.0, w)) * (ks[i] - ks[i - 1]))
    return float("nan")


def variance_extrapolation(
    kappas: Sequence[float], var_ratio: Sequence[float], rise_frac: float = 0.6,
) -> float:
    """Critical-slowing-down extrapolation of the variance divergence.

    For a scalar mode with coefficient phi, the stationary variance scales as
    ``1 / (1 - phi^2)``, so ``1 / var`` falls linearly to zero as ``|phi| -> 1``.
    Fitting a line to ``1 / var_ratio`` over the rising portion of the curve --
    before nonlinear saturation bends it -- and solving for its zero gives a
    threshold-free estimate of the critical gain.  The fit deliberately excludes
    the saturated tail, where the nonlinearity, not the eigenvalue, sets the
    amplitude.
    """
    ks = np.asarray(kappas, dtype=float)
    vr = np.asarray(var_ratio, dtype=float)
    ok = np.isfinite(vr) & (vr > 0)
    ks, vr = ks[ok], vr[ok]
    if ks.size < 4:
        return float("nan")
    order = np.argsort(ks)
    ks, vr = ks[order], vr[order]
    inv = 1.0 / vr
    span = inv[0] - float(np.min(inv))
    if span <= 0:
        return float("nan")
    cutoff = inv[0] - rise_frac * span
    mask = inv >= cutoff
    if mask.sum() < 3:
        mask = np.zeros_like(inv, dtype=bool)
        mask[: max(3, len(inv) // 2)] = True
    slope, intercept = np.polyfit(ks[mask], inv[mask], 1)
    if slope >= 0:
        return float("nan")
    return float(-intercept / slope)


__all__ += ["margin_knee", "variance_extrapolation"]
