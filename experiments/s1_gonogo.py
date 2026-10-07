"""S1 -- causal go/no-go test for b > 0 (Section V-1, Section VI-S1).

The same exact-ground-truth tasks are run under randomised exogenous
background load in two regimes:

*  ``WAIT``              -- deadlines disabled, all planned verification
                            completes.  Pure queueing must raise latency
                            without materially changing correctness.
*  ``RESOURCE_COUPLED``  -- an expired deadline triggers the pre-specified
                            fallback (skipped verification plus a weaker model
                            with error multiplier m).

A third arm disables the shared context store, which sets ``b ~ 0`` *by
construction*: it is the internal negative control, and the load effect must
vanish there even in the resource-coupled regime.

The ECFI pathway is declared supported only if background load degrades
objective correctness materially more in the resource-coupled regime than in
WAIT, with the effect size and test fixed in ``frozen/prereg.md`` before this
script is run.

Usage:  python experiments/s1_gonogo.py [--jobs N] [--seeds N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.runner import RESULTS, default_jobs, run_many, save_json  # noqa: E402
from ecfi.schedules import Schedule  # noqa: E402
from ecfi.sim import RESOURCE_COUPLED, WAIT, SimConfig  # noqa: E402

# Background-load levels, in units of "extra jobs per unit time".  The blocks
# are randomised in order per seed so that load is not confounded with
# simulation time (warm-up drift, store filling).
LOAD_LEVELS = (0.0, 1.0, 2.0, 3.0)
BLOCK_WINDOWS = 40          # windows per load block
KAPPA = 4.0                 # a fixed, stable gain: S1 tests b, not the flip

# Pre-registered in frozen/prereg.md section 2, before any run.  Half of the
# smallest error change considered operationally meaningful (0.01, one extra
# error per 100 tasks).
EFFECT_THRESHOLD = 0.005
ALPHA = 0.05


def load_schedule(levels, rng, windows_per_block, tasks_per_window, lam):
    """Randomised block schedule of exogenous load over simulation time."""
    order = rng.permutation(len(levels))
    block_time = windows_per_block * tasks_per_window / lam
    pts = [(i * block_time, float(levels[j])) for i, j in enumerate(order)]
    return Schedule(pts), [float(levels[j]) for j in order], block_time


def build(seed: int, mode: str, store: bool):
    base = SimConfig(kappa=KAPPA, seed=seed)
    rng = np.random.default_rng(10_000 + seed)
    sched, order, block_time = load_schedule(
        LOAD_LEVELS, rng, BLOCK_WINDOWS, base.W, base.lam()
    )
    cfg = SimConfig(
        kappa=KAPPA,
        mode=mode,
        store_enabled=store,
        bg_load=sched,
        n_windows=BLOCK_WINDOWS * len(LOAD_LEVELS) + 60,
        warmup_windows=60,
        seed=seed,
    )
    return cfg, order, block_time


def per_level(ws, order, block_time):
    """Mean error / latency per background-load level, from window end times."""
    out = {}
    for i, lvl in enumerate(order):
        m = (ws.t_end >= i * block_time) & (ws.t_end < (i + 1) * block_time)
        if m.sum() < 5:
            continue
        out.setdefault(lvl, []).append(
            dict(
                u=float(ws.u[m].mean()),
                wait=float(ws.wait[m].mean()),
                wait_p95=float(ws.wait_p95[m].mean()),
                fallback=float(ws.fallback[m].mean()),
                q=float(ws.q[m].mean()),
                n=int(m.sum()),
            )
        )
    return out


def slope(levels, values):
    """OLS slope of an observable on background-load level."""
    x = np.asarray(levels, float)
    y = np.asarray(values, float)
    if len(x) < 2 or np.ptp(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    arms = [
        ("wait", WAIT, True),
        ("resource_coupled", RESOURCE_COUPLED, True),
        ("resource_coupled_nostore", RESOURCE_COUPLED, False),
    ]

    payload = {"kappa": KAPPA, "load_levels": list(LOAD_LEVELS),
               "block_windows": BLOCK_WINDOWS, "seeds": args.seeds, "arms": {}}

    print(f"S1: {len(arms)} arms x {args.seeds} seeds, "
          f"background-load levels {LOAD_LEVELS}, kappa = {KAPPA}")
    for name, mode, store in arms:
        print(f"\n-- arm: {name}")
        cfgs, orders, bts = [], [], []
        for s in range(1, args.seeds + 1):
            c, o, bt = build(s, mode, store)
            cfgs.append(c); orders.append(o); bts.append(bt)
        series = run_many(cfgs, jobs=args.jobs, label=name)

        # Per-seed statistics are what the pre-registered test needs: the seed
        # is the unit of replication, so pooling windows across seeds first
        # would understate the variance and inflate significance.
        acc = {}
        per_seed_delta = []
        for ws, order, bt in zip(series, orders, bts):
            lv = per_level(ws, order, bt)
            for lvl, recs in lv.items():
                acc.setdefault(lvl, []).extend(recs)
            ls = sorted(lv)
            if len(ls) >= 2:
                lo = float(np.mean([r["u"] for r in lv[ls[0]]]))
                hi = float(np.mean([r["u"] for r in lv[ls[-1]]]))
                per_seed_delta.append(hi - lo)

        levels = sorted(acc)
        stats = {}
        for lvl in levels:
            r = acc[lvl]
            stats[lvl] = {
                k: [float(np.mean([x[k] for x in r])),
                    float(np.std([x[k] for x in r], ddof=1) / np.sqrt(len(r)))]
                for k in ("u", "wait", "wait_p95", "fallback", "q")
            }

        d = np.asarray(per_seed_delta, float)
        payload["arms"][name] = {
            "per_level": stats,
            "slope_u": slope(levels, [stats[l]["u"][0] for l in levels]),
            "slope_wait": slope(levels, [stats[l]["wait"][0] for l in levels]),
            "u_at_min_load": stats[levels[0]]["u"][0],
            "u_at_max_load": stats[levels[-1]]["u"][0],
            "delta_u": float(d.mean()),
            "delta_u_sem": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan"),
            "delta_u_per_seed": [float(x) for x in d],
            "n_seeds": int(d.size),
        }

        print(f"\n== {name}")
        for lvl in levels:
            s = stats[lvl]
            print(f"   bg={lvl:>4.1f}  error={s['u'][0]:.4f}+-{s['u'][1]:.4f}  "
                  f"wait={s['wait'][0]:.3f}  p95={s['wait_p95'][0]:.3f}  "
                  f"fallback={s['fallback'][0]:.3f}")
        arm = payload["arms"][name]
        print(f"   d(error)/d(load) = {arm['slope_u']:+.5f}   "
              f"delta_error = {arm['delta_u']:+.5f} +- {arm['delta_u_sem']:.5f} "
              f"(over {arm['n_seeds']} seeds)")

    # ---- the pre-registered decision rule, executed ----------------------
    a = payload["arms"]
    coupled = np.asarray(a["resource_coupled"]["delta_u_per_seed"], float)
    wait = np.asarray(a["wait"]["delta_u_per_seed"], float)
    effect = float(coupled.mean() - wait.mean())

    t_stat, p_value = ttest_ind(coupled, wait, equal_var=False)  # Welch
    passes_size = effect > EFFECT_THRESHOLD
    passes_sig = float(p_value) < ALPHA
    supported = bool(passes_size and passes_sig)

    payload["verdict"] = {
        "delta_u_wait": float(wait.mean()),
        "delta_u_coupled": float(coupled.mean()),
        "delta_u_nostore": a["resource_coupled_nostore"]["delta_u"],
        "coupled_minus_wait": effect,
        "effect_threshold": EFFECT_THRESHOLD,
        "welch_t": float(t_stat),
        "welch_p": float(p_value),
        "alpha": ALPHA,
        "passes_effect_size": bool(passes_size),
        "passes_significance": bool(passes_sig),
        "ecfi_pathway_supported": supported,
        "propagated_term": effect - (a["resource_coupled_nostore"]["delta_u"]
                                     - float(wait.mean())),
        "direct_term": a["resource_coupled_nostore"]["delta_u"] - float(wait.mean()),
    }

    v = payload["verdict"]
    print("\n== pre-registered decision rule (frozen/prereg.md section 2)")
    print(f"   delta_error  Wait              = {v['delta_u_wait']:+.5f}")
    print(f"   delta_error  Resource-coupled  = {v['delta_u_coupled']:+.5f}")
    print(f"   delta_error  store disabled    = {v['delta_u_nostore']:+.5f}")
    print(f"   effect  (coupled - wait)       = {effect:+.5f}   "
          f"threshold {EFFECT_THRESHOLD}  -> {'PASS' if passes_size else 'FAIL'}")
    print(f"   Welch t-test over seeds        = t {t_stat:+.3f}, p {p_value:.2g}   "
          f"alpha {ALPHA}  -> {'PASS' if passes_sig else 'FAIL'}")
    print(f"\n   b > 0 pathway {'SUPPORTED' if supported else 'NOT SUPPORTED'}")
    if supported:
        print("   S2 may proceed. Commit and tag frozen/ first.")
    else:
        print("   Per the stop condition, S3-S5 are not run as boundary tests;\n"
              "   report the negative result and the conditions map instead.")
    print(f"\n   decomposition of b:  direct (fallback) {v['direct_term']:+.5f}"
          f"   propagated (store) {v['propagated_term']:+.5f}")

    save_json(RESULTS / "s1_gonogo.json", payload)
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
