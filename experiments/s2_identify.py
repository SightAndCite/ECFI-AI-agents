"""S2 -- independent system identification, then freeze (Section V-2, VI-S2).

The six local parameters are estimated from runs driven by three simultaneous,
mutually near-orthogonal excitations:

*  ambiguity bursts        -> move u without moving the policy directly (a, e)
*  background-load pulses  -> move q exogenously, and their decay gives d (b, d)
*  randomized gain dither  -> move v independently of u                 (c, f)

The three use different block periods so that the design matrix is well
conditioned; ``excitation_report`` prints the conditioning, and a run whose
``corr(u, v)`` is close to one has *not* identified anything no matter how good
its R^2 looks.

On success this writes ``frozen/params.json`` and ``frozen/kcrit.json``.  Those
files are the pre-registration: S3 refuses to run without them, and they must
be committed (and tagged) before the sweep.  Re-running this script after the
sweep would destroy the out-of-sample claim, so it refuses to overwrite an
existing freeze unless ``--refreeze`` is passed explicitly.

Usage:  python experiments/s2_identify.py [--jobs N] [--seeds N] [--refreeze]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.identify import identify  # noqa: E402
from ecfi.runner import FROZEN, RESULTS, default_jobs, run_many, save_json  # noqa: E402
from ecfi.schedules import dither, pulse_train  # noqa: E402
from ecfi.sim import SimConfig  # noqa: E402
from regimes import get_regime  # noqa: E402

KAPPA_BASE = 3.0            # a stable operating point; identification is local
V_DITHER_AMP = 0.60         # +/- around nominal effort v_bar
N_WINDOWS = 420
WARMUP = 60
W_IDENT = 400               # wider windows: identification is noise-limited,
                            # and u_k carries binomial noise ~ sqrt(u(1-u)/W)


def build(seed: int, regime) -> SimConfig:
    ov = regime.overrides()
    ref = SimConfig(W=W_IDENT, **ov)
    win_time = ref.W / ref.lam()            # simulation time per window
    horizon = (N_WINDOWS + 5) * win_time
    rng = np.random.default_rng(20_000 + seed)

    # different block periods keep the three excitations from lining up
    bg = pulse_train(base=0.0, high=3.0, period=17 * win_time, duty=0.40,
                     horizon=horizon, phase=3 * win_time)
    amb = pulse_train(base=0.0, high=0.80, period=23 * win_time, duty=0.30,
                      horizon=horizon, phase=11 * win_time)
    vb = dither(base=ref.v_bar, amplitude=V_DITHER_AMP, block=5 * win_time,
                horizon=horizon, rng=rng, lo=0.2)

    return SimConfig(
        **ov,
        kappa=KAPPA_BASE,
        v_bar_sched=vb,
        bg_load=bg,
        ambiguity=amb,
        amb_true_boost=0.25,
        amb_u_boost=0.25,
        W=W_IDENT,
        n_windows=N_WINDOWS,
        warmup_windows=WARMUP,
        seed=seed,
    )


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent), timeout=5,
        )
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unavailable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--bootstrap", type=int, default=600)
    ap.add_argument("--regime", default="shallow")
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite an existing freeze (destroys the out-of-sample claim)")
    args = ap.parse_args()

    regime = get_regime(args.regime)
    out = FROZEN / regime.name
    params_path = out / "params.json"
    kcrit_path = out / "kcrit.json"
    if params_path.exists() and not args.refreeze:
        print(f"refusing to overwrite {params_path}\n"
              f"the freeze is the pre-registration; pass --refreeze only if you are\n"
              f"deliberately restarting the whole identify-predict-sweep sequence.")
        return 1

    print(f"regime {regime.name}: {regime.description}  ({regime.overrides()})")
    cfgs = [build(s, regime) for s in range(1, args.seeds + 1)]
    series = run_many(cfgs, jobs=args.jobs, label="identify")

    res = identify(series, bootstrap=args.bootstrap,
                   rng=np.random.default_rng(7))
    print()
    print(res.summary())

    ok = (res.reg_u.r2 > 0.10 and res.reg_q.r2 > 0.10
          and abs(res.excitation["corr_uv"]) < 0.95)
    if not ok:
        print("\nWARNING: weak fit or poor excitation. Section VI requires this to be\n"
              "reported as a scope limit of the local model, not as support for it.")

    stamp = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "regime": regime.name,
        "regime_overrides": regime.overrides(),
        "kappa_base": KAPPA_BASE,
        "v_dither_amplitude": V_DITHER_AMP,
        "W": W_IDENT,
        "lead_inputs": True,
        "seeds": args.seeds,
        "n_windows": N_WINDOWS,
        "warmup_windows": WARMUP,
        "bootstrap": args.bootstrap,
    }
    save_json(params_path, {**stamp, **res.to_dict()})
    save_json(kcrit_path, {
        **stamp,
        "kappa_crit": res.kappa_crit,
        "kappa_crit_ci": list(res.kappa_crit_ci) if res.kappa_crit_ci else None,
        "kappa_delta": res.kappa_delta,
        "exactness": res.exactness,
        "paradox": res.paradox,
        "sweep_max": 1.5 * res.kappa_crit,
        "note": "frozen before the S3 sweep; do not regenerate afterwards",
    })
    save_json(RESULTS / f"s2_identify_{regime.name}.json", res.to_dict())

    print("\nfrozen. the S3 sweep may now run; it will read these files and "
          "must not be used to revise them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
