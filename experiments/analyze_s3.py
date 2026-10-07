"""Re-derive the S3 endpoints from saved sweep results.

Reads ``results/s3_sweep_<regime>.json`` and applies the threshold-free
endpoints of :mod:`ecfi.detect` to the recorded precursor curves, so the
endpoint definition can be revised without re-running the sweep.  The frozen
prediction is never touched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.detect import margin_knee, variance_extrapolation  # noqa: E402
from ecfi.runner import RESULTS, load_json, save_json  # noqa: E402


def analyze(regime: str) -> dict:
    path = RESULTS / f"s3_sweep_{regime}.json"
    if not path.exists():
        raise SystemExit(f"missing {path}; run experiments/s3_sweep.py --regime {regime}")
    data = load_json(path)
    k_hat = float(data["kappa_hat_crit"])
    out = {"regime": regime, "kappa_hat_crit": k_hat,
           "kappa_hat_ci": data["frozen"].get("kappa_crit_ci"), "by_load": {}}

    print(f"\n### regime {regime}:  frozen kappa_hat_crit = {k_hat:.3f}"
          f"  CI {data['frozen'].get('kappa_crit_ci')}")
    for load, blob in data["by_load"].items():
        pk = blob["per_kappa"]
        ks = sorted(float(k) for k in pk)
        margins = [pk[str(k)]["margin"][0] for k in ks]
        acf = [pk[str(k)]["acf1_u"][0] for k in ks]
        # var_ratio_u is absent from sweeps produced before the endpoint
        # revision; fall back to sd_u where it is available, else skip.
        if "var_ratio_u" in pk[str(ks[0])]:
            varrat = [pk[str(k)]["var_ratio_u"][0] for k in ks]
        elif "sd_u" in pk[str(ks[0])]:
            sd = np.array([pk[str(k)]["sd_u"][0] for k in ks], dtype=float)
            varrat = list((sd / sd[0]) ** 2)
        else:
            varrat = []
        knee = margin_knee(ks, margins)
        vext = variance_extrapolation(ks, varrat) if varrat else float("nan")
        out["by_load"][load] = {
            "margin_knee": knee, "var_extrapolation": vext,
            "ratio_knee": knee / k_hat if np.isfinite(knee) else None,
            "ratio_var": vext / k_hat if np.isfinite(vext) else None,
            "acf1_min": float(np.min(acf)), "margin_min": float(np.min(margins)),
            "var_ratio_max": float(np.max(varrat)) if varrat else None,
        }
        knee_s = f"{knee:6.3f} (ratio {knee / k_hat:.2f})" if np.isfinite(knee) else "   n/a"
        vext_s = f"{vext:6.3f} (ratio {vext / k_hat:.2f})" if np.isfinite(vext) else "   n/a"
        amp_s = f"{np.max(varrat):.1f}x" if varrat else "n/a"
        print(f"  load={load}:  knee={knee_s}   var-extrap={vext_s}   "
              f"min acf1(u)={np.min(acf):+.3f}   min M={np.min(margins):.3f}   "
              f"max var-amp={amp_s}")
    r = [v["ratio_knee"] for v in out["by_load"].values() if v["ratio_knee"]]
    if r:
        out["ratio_knee_mean"] = float(np.mean(r))
        out["ratio_knee_sd"] = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
        print(f"  mean ratio over loads = {np.mean(r):.3f} "
              f"(sd {np.std(r, ddof=1) if len(r) > 1 else 0.0:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="*", default=["shallow", "steep"])
    args = ap.parse_args()
    payload = {r: analyze(r) for r in args.regimes}
    save_json(RESULTS / "s3_endpoints.json", payload)


if __name__ == "__main__":
    main()
