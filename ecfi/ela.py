"""Epistemic load amplification (Section IV).

A strategic participant does not saturate the API and does not compromise a
model; it submits inputs chosen to *look uncertain*, and lets the network's own
"verify more when uncertain" policy convert that into shared-resource demand.

The evaluation quantity is

    A_E = (Delta R / R_base) / (Delta U_inj / U_base),

the normalised, dimensionless form of (9).  ``Delta U_inj`` is the controlled
increment in injected uncertainty and ``Delta R`` the induced increment in
resource demand (verifier effort served, queue occupancy, or tail latency).

The strategic workload is built so that the raw request rate is held fixed:
ambiguity-maximising tasks *replace* benign ones rather than being added, and
by default they raise the agents' uncertainty estimate without raising the
ground-truth error rate (``amb_true_boost = 0``).  That separation is what
distinguishes exploiting the policy from simply sending harder work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .sim import WindowSeries

__all__ = ["ResourceMeasure", "Amplification", "amplification"]

RESOURCE_FIELDS = ("tokens", "q", "wait_p95")


@dataclass
class ResourceMeasure:
    """Baseline and strategic levels of one resource-demand observable."""

    name: str
    base: float
    strategic: float

    @property
    def delta_rel(self) -> float:
        return (self.strategic - self.base) / self.base if self.base > 0 else float("nan")


@dataclass
class Amplification:
    epsilon: float
    u_inj_rel: float
    resources: Dict[str, ResourceMeasure]
    a_e: Dict[str, float]
    error_base: float
    error_strategic: float

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in self.a_e.items())
        return (
            f"eps={self.epsilon:.2f}  dU/U={self.u_inj_rel:+.3f}  A_E: {parts}  "
            f"error {self.error_base:.4f} -> {self.error_strategic:.4f}"
        )


def _mean(ws: WindowSeries, field: str) -> float:
    return float(np.mean(getattr(ws, field)))


def amplification(
    base: WindowSeries,
    strategic: WindowSeries,
    epsilon: float,
    amb_u_boost: float,
    u_field: str = "u",
    inj_field: str = "amb",
) -> Amplification:
    """Normalised epistemic load amplification between two matched runs.

    ``base`` and ``strategic`` must differ only in the strategic-task fraction
    ``epsilon`` and must share a seed, so that the difference is attributable
    to the injected workload rather than to sampling.

    ``Delta U_inj`` is the *apparent* uncertainty actually delivered to the
    policy: the realised strategic-task fraction times the uncertainty each
    such task adds.  It is normalised by the baseline epistemic level, not by
    the baseline verification effort -- normalising by effort would make
    ``A_E`` scale as 1/epsilon and destroy the comparison across injection
    strengths.

    With the congestion-blind law ``v = v_bar + kappa u`` the token-side
    amplification has a closed form, ``A_E(tokens) = kappa u / v``: it exceeds
    one only when the uncertainty-responsive part of verification dominates the
    nominal part.  Queue occupancy and tail latency are convex in load, so
    their amplification can be far larger -- which is the point.  Reporting all
    three makes the difference between "more tokens" and "a congestion event"
    visible instead of averaged away.
    """
    frac = float(np.mean(getattr(strategic, inj_field)) - np.mean(getattr(base, inj_field)))
    # apparent-uncertainty increment actually delivered to the policy
    u_inj_abs = frac * amb_u_boost
    u_base = max(float(np.mean(getattr(base, u_field))), 1e-9)
    u_inj_rel = u_inj_abs / u_base

    resources: Dict[str, ResourceMeasure] = {}
    a_e: Dict[str, float] = {}
    for field in RESOURCE_FIELDS:
        rm = ResourceMeasure(field, _mean(base, field), _mean(strategic, field))
        resources[field] = rm
        a_e[field] = rm.delta_rel / u_inj_rel if u_inj_rel > 0 else float("nan")

    return Amplification(
        epsilon=epsilon,
        u_inj_rel=u_inj_rel,
        resources=resources,
        a_e=a_e,
        error_base=_mean(base, "u"),
        error_strategic=_mean(strategic, "u"),
    )
