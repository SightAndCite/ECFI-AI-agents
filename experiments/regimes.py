"""Operating regimes swept end to end by the benchmark.

The testbed's verification efficacy is a saturating function of effort,
``q_fix(v) = sigma(beta (v - v_half))``.  Its steepness ``beta`` is the single
structural knob that decides whether over-correction can outrun congestion, so
it is the natural axis along which to ask *when the local prediction holds*.

``shallow``  gentle efficacy curve.  A unit of extra effort buys little extra
             correction, so the destabilising over-correction term grows slowly
             with the gain and the congestion path has room to take over first.

``steep``    sharp efficacy curve -- a cheap check catches almost nothing and a
             thorough one catches almost everything.  Over-correction is strong
             and the flip is reachable at a much lower gain.

Running both is the point: a methodology that predicts the transition in one
regime and misses it in the other tells you where the linearization is valid,
which a single well-chosen regime never could.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = ["Regime", "REGIMES", "get_regime"]


@dataclass(frozen=True)
class Regime:
    name: str
    beta: float
    v_half: float
    description: str

    def overrides(self) -> Dict[str, float]:
        return {"beta": self.beta, "v_half": self.v_half, "v_bar": self.v_half}


REGIMES: Dict[str, Regime] = {
    "shallow": Regime(
        "shallow", beta=2.0, v_half=1.30,
        description="gentle verification efficacy; congestion dominates",
    ),
    "steep": Regime(
        "steep", beta=14.0, v_half=1.30,
        description="sharp verification efficacy; over-correction dominates",
    ),
}


def get_regime(name: str) -> Regime:
    if name not in REGIMES:
        raise SystemExit(f"unknown regime {name!r}; choose from {sorted(REGIMES)}")
    return REGIMES[name]
