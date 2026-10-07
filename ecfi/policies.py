"""Verification policies.

Two policies are compared in the benchmark:

* ``ProportionalPolicy``   -- v = sat_[0, v_max](v_bar + kappa * (u_hat - u_ref))
  the uncertainty-proportional law v_k = kappa u_k of the local model, with
  the nonnegativity and saturation that the linearization omits restored.

* ``CongestionAwarePolicy`` -- v = sat_[0, v_max](v_bar + kappa * (u_hat - u_ref) / (1 + gamma * q))
  the cross-layer mitigation of Section V-5, which divides the same gain by
  an observed congestion signal.

Both take the agent's *estimated* uncertainty ``u_hat`` (an imperfectly
calibrated confidence, not the ground-truth error probability) and, for the
congestion-aware law, an observed congestion level ``q_obs``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Policy", "ProportionalPolicy", "CongestionAwarePolicy", "make_policy"]


@dataclass(frozen=True)
class Policy:
    """Base: nominal effort, gain, reference uncertainty, saturation."""

    kappa: float
    v_bar: float = 1.0
    u_ref: float = 0.0
    v_max: float = 4.0

    def effort(self, u_hat: float, q_obs: float = 0.0) -> float:  # pragma: no cover
        raise NotImplementedError

    def _sat(self, v: float) -> float:
        if v < 0.0:
            return 0.0
        if v > self.v_max:
            return self.v_max
        return v


@dataclass(frozen=True)
class ProportionalPolicy(Policy):
    """v = sat(v_bar + kappa * (u_hat - u_ref)); congestion-blind."""

    def effort(self, u_hat: float, q_obs: float = 0.0) -> float:
        return self._sat(self.v_bar + self.kappa * (u_hat - self.u_ref))


@dataclass(frozen=True)
class CongestionAwarePolicy(Policy):
    """v = sat(v_bar + kappa * (u_hat - u_ref) / (1 + gamma * q_obs)).

    gamma = 0 reduces exactly to :class:`ProportionalPolicy`, which is what
    makes the mitigation sweep a clean one-parameter comparison.
    """

    gamma: float = 0.0

    def effort(self, u_hat: float, q_obs: float = 0.0) -> float:
        damp = 1.0 + self.gamma * max(q_obs, 0.0)
        return self._sat(self.v_bar + self.kappa * (u_hat - self.u_ref) / damp)


def make_policy(
    kappa: float,
    gamma: float = 0.0,
    v_bar: float = 1.0,
    u_ref: float = 0.0,
    v_max: float = 4.0,
) -> Policy:
    """Return the congestion-blind policy for gamma == 0, else the aware one."""
    if gamma <= 0.0:
        return ProportionalPolicy(kappa=kappa, v_bar=v_bar, u_ref=u_ref, v_max=v_max)
    return CongestionAwarePolicy(
        kappa=kappa, v_bar=v_bar, u_ref=u_ref, v_max=v_max, gamma=gamma
    )
