"""Local cross-layer linear model, Jury stability conditions, and the
verification-gain limit of Theorem 1.

The model (deviation variables around a nominal operating point):

    u_{k+1} = a u_k + b q_k - c v_k + xi_k
    q_{k+1} = d q_k + e u_k + f v_k + eta_k

with the uncertainty-proportional policy v_k = kappa * u_k, giving

    x_{k+1} = A(kappa) x_k,   A(kappa) = [[a - c*kappa, b],
                                          [e + f*kappa, d]]

Theorem 1.  With b, c, e, f > 0, d > -1, kappa >= 0 and A(0) Schur stable,
every stable uncertainty-proportional policy satisfies

    kappa < kappa_crit = ((1+a)(1+d) - b e) / (c(1+d) + b f).

If additionally 0 <= a, d < 1 and c(1-d) >= b f, the bound is exact and an
eigenvalue equals -1 at kappa = kappa_crit (a flip / period-doubling
bifurcation in the underlying nonlinear system).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

__all__ = [
    "LocalParams",
    "A_matrix",
    "char_poly_coeffs",
    "jury",
    "is_schur",
    "spectral_radius",
    "kappa_crit",
    "kappa_delta",
    "exactness_holds",
    "paradox_holds",
    "u_star",
    "stability_margin",
]


@dataclass(frozen=True)
class LocalParams:
    """The six local coupling parameters.

    a : persistence of epistemic error
    b : congestion -> future epistemic degradation (the ECFI pathway)
    c : direct epistemic benefit of verification
    d : persistence of congestion
    e : compute induced by uncertainty
    f : computational cost of verification
    """

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def validate(self) -> None:
        if not (self.b > 0 and self.c > 0 and self.e > 0 and self.f > 0):
            raise ValueError("Theorem 1 assumes b, c, e, f > 0")
        if self.d <= -1:
            raise ValueError("Theorem 1 assumes d > -1")


def A_matrix(p: LocalParams, kappa: float) -> np.ndarray:
    """Closed-loop system matrix A(kappa)."""
    return np.array(
        [[p.a - p.c * kappa, p.b], [p.e + p.f * kappa, p.d]], dtype=float
    )


def char_poly_coeffs(p: LocalParams, kappa: float):
    """(tau, delta) of p(z) = z^2 - tau z + delta for A(kappa)."""
    tau = p.a - p.c * kappa + p.d
    delta = (p.a - p.c * kappa) * p.d - p.b * (p.e + p.f * kappa)
    return tau, delta


def jury(p: LocalParams, kappa: float):
    """The three second-order Jury quantities (p(1), p(-1), delta).

    Schur stability  <=>  p(1) > 0  and  p(-1) > 0  and  |delta| < 1.
    """
    tau, delta = char_poly_coeffs(p, kappa)
    p1 = 1.0 - tau + delta
    pm1 = 1.0 + tau + delta
    return p1, pm1, delta


def is_schur(p: LocalParams, kappa: float) -> bool:
    p1, pm1, delta = jury(p, kappa)
    return (p1 > 0.0) and (pm1 > 0.0) and (abs(delta) < 1.0)


def spectral_radius(p: LocalParams, kappa: float) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A_matrix(p, kappa)))))


def kappa_crit(p: LocalParams) -> float:
    """Equation (5): the finite verification-gain limit.

    This is the root of p(-1) = 0, i.e. the gain at which an eigenvalue
    reaches -1.  Returns +inf when the denominator vanishes (no oscillatory
    limit) and may be non-positive when the unverified baseline is already
    unstable -- callers should check ``is_schur(p, 0.0)`` first.
    """
    num = (1.0 + p.a) * (1.0 + p.d) - p.b * p.e
    den = p.c * (1.0 + p.d) + p.b * p.f
    if den == 0.0:
        return float("inf")
    return num / den


def kappa_delta(p: LocalParams) -> float:
    """Gain at which the lower determinant condition delta = -1 fails.

    Theorem 1's exactness argument requires kappa_crit < kappa_delta, so
    that p(-1) is the first Jury condition to fail.
    """
    den = p.c * p.d + p.b * p.f
    if den == 0.0:
        return float("inf")
    return (1.0 + p.a * p.d - p.b * p.e) / den


def exactness_holds(p: LocalParams) -> bool:
    """Sufficient condition of Theorem 1 for the bound to be exact."""
    return (0.0 <= p.a < 1.0) and (0.0 <= p.d < 1.0) and (p.c * (1.0 - p.d) >= p.b * p.f)


def paradox_holds(p: LocalParams) -> bool:
    """c(1-d) > b f: more verification improves the static equilibrium."""
    return p.c * (1.0 - p.d) > p.b * p.f


def u_star(p: LocalParams, kappa: float, h: float = 1.0) -> float:
    """Equilibrium uncertainty under a persistent disturbance h.

    u*(kappa) = (1-d) h / p(1).  The denominator is exactly p(1) evaluated
    at A(kappa), which is the spectral form of the verification-congestion
    paradox: raising kappa increases p(1) (static gain margin) while
    decreasing p(-1) (oscillatory margin).
    """
    p1 = (1.0 - p.a) * (1.0 - p.d) - p.b * p.e + kappa * (p.c * (1.0 - p.d) - p.b * p.f)
    if p1 == 0.0:
        return float("inf")
    return (1.0 - p.d) * h / p1


def stability_margin(p: LocalParams, kappa: float) -> float:
    """M = 1 - rho(A(kappa)); the online early-warning signal of Section III."""
    return 1.0 - spectral_radius(p, kappa)


# --- illustrative regime of the paper (analytical illustration only) -------

ILLUSTRATIVE = LocalParams(a=0.6, b=0.3, c=0.8, d=0.6, e=0.4, f=0.6)


if __name__ == "__main__":  # pragma: no cover
    p = ILLUSTRATIVE
    kc = kappa_crit(p)
    print(f"kappa_crit      = {kc:.6f}")
    print(f"kappa_delta     = {kappa_delta(p):.6f}")
    print(f"exactness holds = {exactness_holds(p)}")
    print(f"paradox holds   = {paradox_holds(p)}")
    print(f"rho at kappa_crit = {spectral_radius(p, kc):.6f}")
    print(f"rho at kappa=1.7  = {spectral_radius(p, 1.7):.6f}")
    print(f"u*(0)/u*(kc)      = {u_star(p, 0.0) / u_star(p, kc):.4f}")
