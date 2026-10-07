"""S5 -- cross-layer mitigation (Section V-5, VI-S5).

Compares the congestion-blind law ``v = kappa u`` against the congestion-aware
law

    v = kappa u / (1 + gamma q),     gamma >= 0,

sweeping ``gamma`` at a gain *beyond* the transition located in S3, where the
blind policy is already oscillating.

The target is not minimum verification.  It is a better
quality-latency-cost-stability frontier, so all four axes are reported
together: objective error (quality), p95 verification wait (latency), verifier
effort served (cost), and the online stability margin plus the alternation
amplitude (stability).  A gamma that buys stability by simply verifying less
should be visible as a quality loss in the same table.

Usage:  python experiments/s5_mitigation.py [--jobs N] [--seeds N] [--regime R]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.detect import (  # noqa: E402
    alternation_amplitude, lag1_autocorr, nyquist_fraction, online_margin,
)
from ecfi.runner import FROZEN, RESULTS, default_jobs, load_json, run_many, save_json  # noqa: E402
from ecfi.sim import SimConfig  # noqa: E402
from regimes import get_regime  # noqa: E402

# q is measured as waiting requests per verifier server and sits well below 1
# at moderate load, so gamma must reach tens before 1 + gamma q damps anything
GAMMAS = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0)
GAIN_MULTIPLE = 1.25        # operate beyond the frozen boundary


def summarize(ws) -> dict:
    m = online_margin(ws.u, ws.q, ws.v)
    return dict(
        error=float(ws.u.mean()),
        wait_p95=float(ws.wait_p95.mean()),
        cost=float(ws.tokens.mean()),
        v=float(ws.v.mean()),
        q=float(ws.q.mean()),
        fallback=float(ws.fallback.mean()),
        margin=float(np.nanmean(m)) if np.any(np.isfinite(m)) else float("nan"),
        acf1_u=lag1_autocorr(ws.u),
        nyq_v=nyquist_fraction(ws.v),
        amp_v=alternation_amplitude(ws.v),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--regime", default="steep")
    ap.add_argument("--load", type=float, default=0.65)
    args = ap.parse_args()

    regime = get_regime(args.regime)
    ov = regime.overrides()
    kc = load_json(FROZEN / regime.name / "kcrit.json")
    k_hat = float(kc["kappa_crit"])
    kappa = GAIN_MULTIPLE * k_hat

    print(f"regime {regime.name}: kappa = {GAIN_MULTIPLE} x kappa_hat_crit "
          f"= {kappa:.3f}, load_factor = {args.load}")

    payload = {"regime": regime.name, "kappa": kappa, "kappa_hat_crit": k_hat,
               "gammas": list(GAMMAS), "load_factor": args.load,
               "seeds": args.seeds, "by_gamma": {}}

    print("\n  gamma    error   p95wait    cost     v      q     fb    "
          "margin  acf1(u)  amp(v)")
    for g in GAMMAS:
        cfgs = [SimConfig(**ov, kappa=kappa, gamma=g, load_factor=args.load,
                          n_windows=args.windows, warmup_windows=60, seed=s)
                for s in range(1, args.seeds + 1)]
        series = run_many(cfgs, jobs=args.jobs, progress=False)
        rows = [summarize(ws) for ws in series]
        agg = {k: [float(np.mean([r[k] for r in rows])),
                   float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows)))]
               for k in rows[0]}
        payload["by_gamma"][str(g)] = agg
        print(f"  {g:5.1f}   {agg['error'][0]:.4f}   {agg['wait_p95'][0]:6.3f}  "
              f"{agg['cost'][0]:7.1f}  {agg['v'][0]:.3f}  {agg['q'][0]:.3f}  "
              f"{agg['fallback'][0]:.3f}  {agg['margin'][0]:+.3f}  "
              f"{agg['acf1_u'][0]:+.3f}  {agg['amp_v'][0]:.4f}")

    # a gamma is only an improvement if it buys stability without losing quality
    base = payload["by_gamma"]["0.0"]
    best, best_g = None, None
    for g in GAMMAS[1:]:
        a = payload["by_gamma"][str(g)]
        if a["error"][0] <= base["error"][0] * 1.02 and a["margin"][0] > base["margin"][0]:
            score = a["margin"][0] - base["margin"][0]
            if best is None or score > best:
                best, best_g = score, g
    payload["best_gamma"] = best_g
    print(f"\n  best gamma (stability gain at no more than 2% quality loss): {best_g}")

    save_json(RESULTS / f"s5_mitigation_{regime.name}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
