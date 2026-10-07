"""Calibration scan: find a regime in which the nonlinear testbed actually
exhibits a reachable flip, without the congestion path overloading first.

This is *model construction*, not a result.  It runs before anything is
frozen: the purpose is to choose a testbed whose ground truth is controllable
and whose instability is reachable, so that the identify-then-predict
methodology of Section VI has something to predict.  Nothing here is used to
fit kappa_crit.
"""

from __future__ import annotations

import itertools
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/home/claude/ecfi")
from ecfi.sim import SimConfig, run_sim  # noqa: E402


def acf1(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    return float(np.dot(x[:-1], x[1:]) / max(np.dot(x, x), 1e-12))


def nyq_frac(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    return float(P[-1] / max(P[1:].sum(), 1e-12))


def one(args):
    beta, v_half, delta_dep, load, kappa, seed = args
    cfg = SimConfig(
        kappa=kappa, beta=beta, v_half=v_half, v_bar=v_half,
        delta_dep=delta_dep, load_factor=load,
        deadline=2.0, W=200, n_windows=260, warmup_windows=60, seed=seed,
    )
    ws = run_sim(cfg)
    return dict(
        beta=beta, v_half=v_half, delta_dep=delta_dep, load=load,
        kappa=kappa, seed=seed,
        u=ws.u.mean(), v=ws.v.mean(), q=ws.q.mean(), fb=ws.fallback.mean(),
        sdv=ws.v.std(), acf_u=acf1(ws.u), nyq_v=nyq_frac(ws.v),
        nyq_u=nyq_frac(ws.u),
    )


def main():
    betas = [14.0, 22.0]
    v_halves = [1.3]
    deltas = [0.6, 0.85]
    loads = [0.55]
    kappas = [0.0, 3.0, 6.0, 9.0, 12.0, 16.0, 20.0]
    seeds = [1, 2, 3]

    grid = list(itertools.product(betas, v_halves, deltas, loads, kappas, seeds))
    with Pool(2) as pool:
        rows = pool.map(one, grid, chunksize=4)

    keys = ("beta", "v_half", "delta_dep", "load")
    for combo in itertools.product(betas, v_halves, deltas, loads):
        sel = [r for r in rows if tuple(r[k] for k in keys) == combo]
        print(f"\n=== beta={combo[0]} v_half={combo[1]} delta={combo[2]} load={combo[3]}")
        for k in kappas:
            s = [r for r in sel if r["kappa"] == k]
            f = lambda n: np.mean([r[n] for r in s])  # noqa: E731
            print(
                f"  k={k:>5.1f} u={f('u'):.4f} v={f('v'):.3f} q={f('q'):.3f} "
                f"fb={f('fb'):.3f} sd(v)={f('sdv'):.4f} "
                f"acf1_u={f('acf_u'):+.3f} nyq_u={f('nyq_u'):.3f} nyq_v={f('nyq_v'):.3f}"
            )


if __name__ == "__main__":
    main()
