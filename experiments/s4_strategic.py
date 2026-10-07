"""S4 -- strategic-pressure condition and epistemic load amplification (V-4, VI-S4).

A fraction ``epsilon`` of benign tasks is *replaced* by ambiguity-maximising
inputs, holding the raw request rate fixed.  By default those inputs raise the
agents' apparent uncertainty without raising ground-truth difficulty
(``amb_true_boost = 0``): the attacker is exploiting the *policy*, not sending
harder work.  That separation is what distinguishes epistemic load
amplification from an ordinary load increase, and it is why ``A_E`` measures
the gain of the control law rather than the cost of an input.

The question of interest is whether ``A_E`` and downstream task error rise
sharply as the operating gain approaches the transition located in S3.

Usage:  python experiments/s4_strategic.py [--jobs N] [--seeds N] [--regime R]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.ela import amplification  # noqa: E402
from ecfi.runner import FROZEN, RESULTS, default_jobs, load_json, run_many, save_json  # noqa: E402
from ecfi.schedules import constant  # noqa: E402
from ecfi.sim import SimConfig  # noqa: E402
from regimes import get_regime  # noqa: E402

EPSILONS = (0.0, 0.05, 0.10, 0.20)
GAIN_FRACTIONS = (0.25, 0.50, 0.75, 1.00)   # of the frozen kappa_hat_crit
AMB_U_BOOST = 0.25                          # apparent uncertainty per strategic task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--regime", default="steep")
    ap.add_argument("--true-boost", type=float, default=0.0,
                    help="ground-truth difficulty added by a strategic task; "
                         "0 isolates pure policy exploitation")
    args = ap.parse_args()

    regime = get_regime(args.regime)
    ov = regime.overrides()
    kc = load_json(FROZEN / regime.name / "kcrit.json")
    k_hat = float(kc["kappa_crit"])
    gains = [g * k_hat for g in GAIN_FRACTIONS]

    print(f"regime {regime.name}: frozen kappa_hat_crit = {k_hat:.3f}")
    print(f"gains {['%.2f' % g for g in gains]}, epsilons {EPSILONS}, "
          f"amb_true_boost = {args.true_boost}")

    def cfg(kappa, eps, seed):
        return SimConfig(
            **ov, kappa=kappa, seed=seed,
            ambiguity=constant(eps) if eps > 0 else None,
            amb_true_boost=args.true_boost, amb_u_boost=AMB_U_BOOST,
            n_windows=args.windows, warmup_windows=60,
        )

    payload = {"regime": regime.name, "kappa_hat_crit": k_hat,
               "epsilons": list(EPSILONS), "gain_fractions": list(GAIN_FRACTIONS),
               "amb_true_boost": args.true_boost, "amb_u_boost": AMB_U_BOOST,
               "by_gain": {}}

    for gf, kappa in zip(GAIN_FRACTIONS, gains):
        runs = {}
        for eps in EPSILONS:
            cfgs = [cfg(kappa, eps, s) for s in range(1, args.seeds + 1)]
            runs[eps] = run_many(cfgs, jobs=args.jobs,
                                 label=f"k={kappa:.2f} eps={eps}")

        rows = {}
        print(f"\n== kappa = {kappa:.3f}  ({gf:.0%} of kappa_hat_crit)")
        for eps in EPSILONS[1:]:
            # seed-matched pairs: the only difference is the injected workload
            amps = [amplification(b, s, eps, AMB_U_BOOST)
                    for b, s in zip(runs[0.0], runs[eps])]
            rows[eps] = {
                "a_e": {f: [float(np.mean([a.a_e[f] for a in amps])),
                            float(np.std([a.a_e[f] for a in amps], ddof=1)
                                  / np.sqrt(len(amps)))]
                        for f in amps[0].a_e},
                "error_base": float(np.mean([a.error_base for a in amps])),
                "error_strategic": float(np.mean([a.error_strategic for a in amps])),
                "u_inj_rel": float(np.mean([a.u_inj_rel for a in amps])),
            }
            r = rows[eps]
            print(f"   eps={eps:.2f}  A_E(tokens)={r['a_e']['tokens'][0]:+.3f}"
                  f"  A_E(queue)={r['a_e']['q'][0]:+.3f}"
                  f"  A_E(p95)={r['a_e']['wait_p95'][0]:+.3f}"
                  f"   error {r['error_base']:.4f} -> {r['error_strategic']:.4f}")

        payload["by_gain"][f"{gf:.2f}"] = {"kappa": kappa, "by_epsilon": rows}

    save_json(RESULTS / f"s4_strategic_{regime.name}.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
