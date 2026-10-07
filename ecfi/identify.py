"""Independent system identification of the local parameters (Section V-2).

The local model is fitted to windowed observables in *deviation* form,

    u_{k+1} = a u_k + b q_k - c v_k + xi_k
    q_{k+1} = d q_k + e u_k + f v_k + eta_k,

by ordinary least squares, one equation at a time.  Two things matter more
than the point estimates:

*  **Excitation.**  Least squares on a closed-loop trace with no exogenous
   input is not identification -- it recovers the closed-loop map, not the
   open-loop coefficients.  Every estimate here must come from a run driven by
   the targeted interventions of :mod:`ecfi.schedules` (ambiguity bursts,
   background-load pulses, gain dithering), and :func:`excitation_report`
   reports the conditioning that makes the estimate meaningful.

*  **Honest intervals.**  The residuals are autocorrelated, so i.i.d.
   bootstrapping would understate the uncertainty.  :func:`bootstrap_params`
   uses a moving-block bootstrap whose block length is set by the residual
   correlation time.

Nothing in this module is allowed to see the sweep of Section V-3: the
estimates and the kappa_crit derived from them are frozen first.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linear import LocalParams, kappa_crit, kappa_delta, exactness_holds, paradox_holds
from .sim import WindowSeries

__all__ = [
    "Regression",
    "IdentificationResult",
    "build_design",
    "identify",
    "bootstrap_params",
    "excitation_report",
]


# --------------------------------------------------------------------------- #
# ordinary least squares with diagnostics
# --------------------------------------------------------------------------- #


@dataclass
class Regression:
    coef: np.ndarray
    names: Tuple[str, ...]
    r2: float
    r2_adj: float
    resid: np.ndarray
    cond: float
    n: int

    def as_dict(self) -> Dict:
        return {
            "coef": {n: float(c) for n, c in zip(self.names, self.coef)},
            "r2": self.r2,
            "r2_adj": self.r2_adj,
            "cond": self.cond,
            "n": self.n,
        }


def _ols(X: np.ndarray, y: np.ndarray, names: Sequence[str]) -> Regression:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, p = X.shape
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / max(n - p, 1) if np.isfinite(r2) else float("nan")
    cond = float(np.linalg.cond(X))
    return Regression(coef, tuple(names), r2, r2_adj, resid, cond, n)


# --------------------------------------------------------------------------- #
# design matrices
# --------------------------------------------------------------------------- #


def build_design(series: Sequence[WindowSeries], lead_inputs: bool = True):
    """Stack one-step-ahead regressors across runs, in deviation form.

    Each run contributes rows k = 0 .. K-2; rows never cross a run boundary,
    so concatenating independent replications is safe.  Deviations are taken
    per run, which removes each run's own operating point -- exactly the
    linearization the local model assumes.

    **Indexing convention.**  Section II writes ``v_k = kappa u_k``: the effort
    indexed k is the response to the uncertainty indexed k, and it acts on the
    transition into k+1.  In the testbed the policy observes the monitoring
    signal of a closed window and applies the resulting effort during the
    *next* window, so the physically recorded ``v`` and ``q`` series are one
    index ahead of the paper's convention.  ``lead_inputs=True`` (the default)
    restores it by regressing

        u_{k+1} ~ [u_k, q_{k+1}, v_{k+1}]
        q_{k+1} ~ [q_k, u_k,     v_{k+1}]

    which makes the closed loop identical to A(kappa): with v_{k+1} = kappa u_k,
    the u-equation collapses to u_{k+1} = (a - c kappa) u_k + b q_{k+1}.
    Setting ``lead_inputs=False`` uses the raw recorded indices; it yields a
    near-zero (and sign-unstable) estimate of ``c``, because the contemporaneous
    effect of effort on error has already been absorbed into u_k.  Which
    convention a deployment needs depends on where its monitoring sits relative
    to its actuation, and it must be stated, not assumed.
    """
    Xu: List[np.ndarray] = []
    yu: List[np.ndarray] = []
    Xq: List[np.ndarray] = []
    yq: List[np.ndarray] = []

    for s in series:
        u, q, v = s.deviations()
        if len(u) < 3:
            continue
        if lead_inputs:
            Xu.append(np.column_stack([u[:-1], q[1:], v[1:]]))
            Xq.append(np.column_stack([q[:-1], u[:-1], v[1:]]))
        else:
            Xu.append(np.column_stack([u[:-1], q[:-1], v[:-1]]))
            Xq.append(np.column_stack([q[:-1], u[:-1], v[:-1]]))
        yu.append(u[1:])
        yq.append(q[1:])

    if not Xu:
        raise ValueError("no usable windows for identification")
    return (
        np.vstack(Xu), np.concatenate(yu),
        np.vstack(Xq), np.concatenate(yq),
    )


# --------------------------------------------------------------------------- #
# identification
# --------------------------------------------------------------------------- #


@dataclass
class IdentificationResult:
    params: LocalParams
    kappa_crit: float
    kappa_delta: float
    exactness: bool
    paradox: bool
    reg_u: Regression
    reg_q: Regression
    ci: Optional[Dict[str, Tuple[float, float]]] = None
    kappa_crit_ci: Optional[Tuple[float, float]] = None
    excitation: Optional[Dict[str, float]] = None

    def summary(self) -> str:
        p = self.params
        lines = [
            "identified local parameters",
            f"  a = {p.a:+.4f}   b = {p.b:+.4f}   c = {p.c:+.4f}",
            f"  d = {p.d:+.4f}   e = {p.e:+.4f}   f = {p.f:+.4f}",
            f"  R2(u) = {self.reg_u.r2:.3f}   R2(q) = {self.reg_q.r2:.3f}",
            f"  kappa_crit = {self.kappa_crit:.4f}   kappa_delta = {self.kappa_delta:.4f}",
            f"  exactness condition c(1-d) >= b f : {self.exactness}",
            f"  paradox condition   c(1-d) >  b f : {self.paradox}",
        ]
        if self.kappa_crit_ci is not None:
            lo, hi = self.kappa_crit_ci
            lines.append(f"  kappa_crit 95% CI = [{lo:.4f}, {hi:.4f}]")
        if self.excitation is not None:
            lines.append(
                "  excitation: cond(u-eq) = {cu:.1f}, cond(q-eq) = {cq:.1f}, "
                "corr(u,v) = {r:.3f}".format(
                    cu=self.excitation["cond_u"],
                    cq=self.excitation["cond_q"],
                    r=self.excitation["corr_uv"],
                )
            )
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        d = {
            "params": self.params.as_dict(),
            "kappa_crit": self.kappa_crit,
            "kappa_delta": self.kappa_delta,
            "exactness": bool(self.exactness),
            "paradox": bool(self.paradox),
            "reg_u": self.reg_u.as_dict(),
            "reg_q": self.reg_q.as_dict(),
        }
        if self.ci is not None:
            d["ci"] = {k: [float(v[0]), float(v[1])] for k, v in self.ci.items()}
        if self.kappa_crit_ci is not None:
            d["kappa_crit_ci"] = [float(self.kappa_crit_ci[0]), float(self.kappa_crit_ci[1])]
        if self.excitation is not None:
            d["excitation"] = self.excitation
        return d


def _params_from_coefs(cu: np.ndarray, cq: np.ndarray) -> LocalParams:
    # u-equation regressors: [u, q, v] -> [a, b, -c]
    # q-equation regressors: [q, u, v] -> [d, e,  f]
    return LocalParams(
        a=float(cu[0]), b=float(cu[1]), c=float(-cu[2]),
        d=float(cq[0]), e=float(cq[1]), f=float(cq[2]),
    )


def identify(series: Sequence[WindowSeries], bootstrap: int = 0,
             rng: Optional[np.random.Generator] = None,
             lead_inputs: bool = True) -> IdentificationResult:
    """Fit the local parameters and derive kappa_crit from them."""
    Xu, yu, Xq, yq = build_design(series, lead_inputs=lead_inputs)
    reg_u = _ols(Xu, yu, ("u", "q", "v"))
    reg_q = _ols(Xq, yq, ("q", "u", "v"))
    params = _params_from_coefs(reg_u.coef, reg_q.coef)

    res = IdentificationResult(
        params=params,
        kappa_crit=kappa_crit(params),
        kappa_delta=kappa_delta(params),
        exactness=exactness_holds(params),
        paradox=paradox_holds(params),
        reg_u=reg_u,
        reg_q=reg_q,
        excitation=excitation_report(Xu, Xq),
    )

    if bootstrap > 0:
        ci, kc_ci = bootstrap_params(series, n_boot=bootstrap, rng=rng,
                                     lead_inputs=lead_inputs)
        res.ci = ci
        res.kappa_crit_ci = kc_ci
    return res


# --------------------------------------------------------------------------- #
# bootstrap and excitation diagnostics
# --------------------------------------------------------------------------- #


def _block_length(resid: np.ndarray) -> int:
    """Crude but honest block length: 2x the residual correlation time."""
    r = resid - resid.mean()
    denom = float(r @ r)
    if denom <= 0:
        return 4
    tau = 1
    for lag in range(1, min(25, len(r) // 4)):
        rho = float(r[:-lag] @ r[lag:]) / denom
        if rho < 1.0 / np.e:
            tau = lag
            break
        tau = lag
    return int(max(4, 2 * tau))


def bootstrap_params(
    series: Sequence[WindowSeries],
    n_boot: int = 400,
    rng: Optional[np.random.Generator] = None,
    alpha: float = 0.05,
    lead_inputs: bool = True,
):
    """Moving-block bootstrap over the stacked design rows.

    Returns ``(ci, kappa_crit_ci)`` where ``ci`` maps each of a..f to a
    percentile interval.
    """
    rng = rng or np.random.default_rng(0)
    Xu, yu, Xq, yq = build_design(series, lead_inputs=lead_inputs)
    reg_u = _ols(Xu, yu, ("u", "q", "v"))
    reg_q = _ols(Xq, yq, ("q", "u", "v"))
    L = max(_block_length(reg_u.resid), _block_length(reg_q.resid))

    n = Xu.shape[0]
    n_blocks = int(np.ceil(n / L))
    draws: List[LocalParams] = []
    kcs: List[float] = []

    for _ in range(n_boot):
        starts = rng.integers(0, max(n - L, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, min(s + L, n)) for s in starts])[:n]
        try:
            cu, *_ = np.linalg.lstsq(Xu[idx], yu[idx], rcond=None)
            cq, *_ = np.linalg.lstsq(Xq[idx], yq[idx], rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover
            continue
        p = _params_from_coefs(cu, cq)
        draws.append(p)
        kcs.append(kappa_crit(p))

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    ci = {}
    for name in ("a", "b", "c", "d", "e", "f"):
        vals = np.array([getattr(p, name) for p in draws], dtype=float)
        ci[name] = (float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q)))
    kcs_arr = np.array([k for k in kcs if np.isfinite(k)], dtype=float)
    kc_ci = (
        float(np.percentile(kcs_arr, lo_q)),
        float(np.percentile(kcs_arr, hi_q)),
    ) if kcs_arr.size else (float("nan"), float("nan"))
    return ci, kc_ci


def excitation_report(Xu: np.ndarray, Xq: np.ndarray) -> Dict[str, float]:
    """Persistent-excitation diagnostics.

    A near-collinear design (very large condition number, or |corr(u, v)| close
    to one) means the run did not separate the policy from the plant: the fit
    may be good and the coefficients still meaningless.  Report it, do not hide
    it behind R^2.
    """
    u, v = Xu[:, 0], Xu[:, 2]
    denom = float(np.std(u) * np.std(v))
    corr_uv = float(np.mean((u - u.mean()) * (v - v.mean())) / denom) if denom > 0 else float("nan")
    return {
        "cond_u": float(np.linalg.cond(Xu)),
        "cond_q": float(np.linalg.cond(Xq)),
        "corr_uv": corr_uv,
        "n_rows": int(Xu.shape[0]),
    }
