"""Piecewise-constant exogenous schedules.

Identification (Section V-2) needs *targeted interventions*: background-load
pulses, ambiguity bursts, and randomized dithering of the verification gain.
All three are expressed here as piecewise-constant functions of simulation
time, so that an experiment can declare its excitation up front and the
simulator stays oblivious to which parameter is being identified.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["Schedule", "constant", "pulse_train", "dither"]


class Schedule:
    """A right-continuous piecewise-constant function of time."""

    __slots__ = ("_t", "_v")

    def __init__(self, breakpoints: Iterable[Tuple[float, float]]):
        pts: List[Tuple[float, float]] = sorted(breakpoints, key=lambda p: p[0])
        if not pts:
            raise ValueError("a schedule needs at least one breakpoint")
        if pts[0][0] > 0.0:
            pts.insert(0, (0.0, pts[0][1]))
        self._t = [p[0] for p in pts]
        self._v = [p[1] for p in pts]

    def at(self, t: float) -> float:
        i = bisect_right(self._t, t) - 1
        if i < 0:
            i = 0
        return self._v[i]

    @property
    def breakpoints(self) -> Sequence[Tuple[float, float]]:
        return list(zip(self._t, self._v))

    def __repr__(self) -> str:  # pragma: no cover
        return f"Schedule({self.breakpoints!r})"


def constant(value: float) -> Schedule:
    return Schedule([(0.0, value)])


def pulse_train(
    base: float,
    high: float,
    period: float,
    duty: float,
    horizon: float,
    phase: float = 0.0,
) -> Schedule:
    """Square-wave excitation: ``high`` for ``duty*period``, else ``base``."""
    if not (0.0 < duty < 1.0):
        raise ValueError("duty must lie in (0, 1)")
    pts: List[Tuple[float, float]] = [(0.0, base)]
    t = phase
    while t < horizon:
        pts.append((t, high))
        pts.append((t + duty * period, base))
        t += period
    return Schedule(pts)


def dither(
    base: float,
    amplitude: float,
    block: float,
    horizon: float,
    rng: np.random.Generator,
    lo: float = 0.0,
) -> Schedule:
    """Randomized block dithering, the excitation used to identify c and f.

    The value is redrawn uniformly from ``base +/- amplitude`` every ``block``
    time units.  Blockwise (rather than per-task) dithering keeps the windowed
    observables informative: a gain that changes faster than the window length
    would be averaged away before identification sees it.
    """
    pts: List[Tuple[float, float]] = []
    t = 0.0
    while t < horizon:
        val = base + rng.uniform(-amplitude, amplitude)
        pts.append((t, max(lo, val)))
        t += block
    if not pts:
        pts = [(0.0, base)]
    return Schedule(pts)
