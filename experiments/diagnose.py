"""Environment diagnostic — run this first when anything behaves oddly.

Most "impossible" failures in a Python project of this shape are install
problems, not logic problems: a second copy of the package earlier on
``sys.path``, an editable install pointing somewhere else, a stale
``__pycache__``, or a dependency at a version the code was never run against.
This script prints exactly what is loaded from where, checksums the modules, and
then runs one simulation serially and one in a worker process.

Usage:  python experiments/diagnose.py
"""

from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return "unreadable"


def find_duplicate_packages(name: str = "ecfi"):
    """Every importable copy of the package on sys.path, in resolution order."""
    hits = []
    for entry in sys.path:
        if not entry:
            entry = os.getcwd()
        p = Path(entry) / name / "__init__.py"
        if p.exists():
            hits.append(p.resolve())
    seen, uniq = set(), []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq


def main() -> int:
    print("=" * 68)
    print("ECFI environment diagnostic")
    print("=" * 68)

    print(f"\npython      {sys.version.split()[0]}  ({platform.python_implementation()})")
    print(f"platform    {platform.platform()}")
    print(f"executable  {sys.executable}")
    print(f"cwd         {os.getcwd()}")
    try:
        import multiprocessing as mp
        print(f"mp start    {mp.get_start_method()}")
    except Exception as e:
        print(f"mp start    unavailable ({e})")

    print("\ndependencies")
    for mod in ("numpy", "scipy", "simpy", "matplotlib", "sympy"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<12} {getattr(m, '__version__', '?'):<10} {m.__file__}")
        except Exception as e:
            print(f"  {mod:<12} MISSING ({e})")

    print("\necfi package")
    copies = find_duplicate_packages()
    if len(copies) > 1:
        print("  !! more than one copy of `ecfi` is importable; the first wins:")
        for c in copies:
            print(f"     {c}")
        print("     Remove or rename the others, or the file you edit is not the "
              "file that runs.")
    try:
        import ecfi
        pkg = Path(ecfi.__file__).parent
        print(f"  imported from {pkg}")
        print(f"  version       {getattr(ecfi, '__version__', '?')}")
        for f in ("sim.py", "runner.py", "linear.py", "identify.py", "detect.py"):
            fp = pkg / f
            if fp.exists():
                print(f"    {f:<14} sha256:{sha(fp)}  {fp.stat().st_size:>6} bytes")
            else:
                print(f"    {f:<14} MISSING")
    except Exception:
        print("  could not import ecfi:")
        traceback.print_exc()
        return 1

    caches = list(pkg.rglob("__pycache__"))
    if caches:
        print(f"\n  {len(caches)} __pycache__ director(ies) present. If this install was "
              "updated\n  in place, delete them and retry before reading anything else "
              "into a failure:")
        print("    Windows : Get-ChildItem -Recurse -Directory __pycache__ | "
              "Remove-Item -Recurse -Force")
        print("    Unix    : find . -name __pycache__ -type d -exec rm -rf {} +")

    # --- functional checks -------------------------------------------------
    print("\nfunctional checks")
    from ecfi.sim import SimConfig, run_sim  # noqa: E402

    cfg = SimConfig(kappa=2.0, W=50, n_windows=40, warmup_windows=10, seed=1)
    try:
        import simpy
        from ecfi.sim import _Sim
        probe = _Sim(cfg)
        print(f"  verifier pool type   {type(probe.pool).__module__}."
              f"{type(probe.pool).__name__}"
              f"   (expected simpy.resources.resource.Resource)")
        print(f"  pool capacity        {probe.pool.capacity}")
        assert isinstance(probe.pool, simpy.Resource)
    except Exception:
        print("  FAILED while constructing the simulator:")
        traceback.print_exc()
        return 1

    try:
        ws = run_sim(cfg)
        print(f"  serial run           OK, {len(ws)} windows, "
              f"mean error {ws.u.mean():.4f}")
    except Exception:
        print("  FAILED in a serial run — this is a logic or install problem, "
              "not a parallelism one:")
        traceback.print_exc()
        return 1

    from ecfi.runner import default_jobs, run_many

    try:
        out = run_many([cfg, cfg], jobs=2, progress=False)
        print(f"  parallel run (2)     OK, {len(out)} results")
    except Exception:
        print("  FAILED with 2 workers — the simulation itself is fine, "
              "so suspect pickling or the start method:")
        traceback.print_exc()
        return 1

    # The failure mode that matters here only appears under real concurrency:
    # many fresh interpreters importing the package at the same moment.  Two
    # workers never reproduce it, so push the width up to what the machine has.
    wide = min(default_jobs(), 32)
    if wide > 2:
        try:
            out = run_many([cfg] * wide, jobs=wide, progress=False)
            print(f"  parallel run ({wide:>2})    OK, {len(out)} results")
        except Exception as e:
            print(f"  FAILED at {wide} workers but passed at 2 — this is a "
                  "concurrency-triggered\n  environment fault, not a logic "
                  "error. Details:")
            print(f"    {type(e).__name__}: {e}")
            return 1

    print("\nall checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
