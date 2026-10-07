"""Parallel execution helpers and result I/O.

Replications and sweep cells are independent, so the only parallelism that
matters is across runs.  The simulation itself is a single-threaded event loop:
it is CPU-bound and uses no GPU, so wall-clock scales with the number of cores
you give ``--jobs``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import is_dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .sim import SimConfig, WindowSeries, run_sim

__all__ = [
    "ROOT", "RESULTS", "FROZEN", "FIGS",
    "available_cpus", "default_jobs", "run_many", "EnvironmentFault",
    "save_json", "load_json", "jsonable",
]

# Line-buffer stdout for every script in this package: a long run redirected to
# a file or piped into `tee` would otherwise show nothing until it ends.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FROZEN = ROOT / "frozen"
FIGS = ROOT / "figs"
for _d in (RESULTS, FROZEN, FIGS):
    _d.mkdir(exist_ok=True)


def available_cpus() -> int:
    """CPUs this process may actually use.

    ``os.cpu_count()`` reports the *machine's* CPUs, which is the wrong number
    in a container, a cgroup, or under a batch scheduler: a job pinned to 4 CPUs
    on a 96-core node would otherwise start 96 workers and thrash.
    ``sched_getaffinity`` reports the allowed set and is honoured by Docker,
    Slurm, taskset and cpusets, so it is preferred where available.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))  # Linux
    except AttributeError:
        return max(1, os.cpu_count() or 1)           # macOS, Windows


def default_jobs() -> int:
    """Default worker count: every CPU this process is allowed to use.

    An explicit ``--jobs`` always wins.  On an SMT ("hyper-threaded") machine
    this counts logical CPUs; for this workload -- pure-Python event loops --
    the sibling threads add perhaps 10-20%, not 2x, so setting ``--jobs`` to the
    *physical* core count costs little and leaves the machine usable.
    """
    return available_cpus()


class EnvironmentFault(RuntimeError):
    """A worker failed for a reason outside the simulation's control."""


# Signatures of a corrupt or partially written .pyc being executed.  These are
# environment faults -- the source is fine and the same run succeeds serially.
_BYTECODE_FAULTS = ("unknown opcode", "bad marshal", "bad magic number",
                    "unsupported pyc", "truncated")


def _is_bytecode_fault(exc: BaseException) -> bool:
    if isinstance(exc, (SystemError, EOFError, ImportError, ValueError)):
        msg = str(exc).lower()
        return any(s in msg for s in _BYTECODE_FAULTS)
    return False


def _prepare_worker_cache() -> None:
    """Take this package's bytecode cache out of the picture for workers.

    Under ``spawn`` every worker is a fresh interpreter that imports this
    package again.  If a worker reads a ``.pyc`` that is incomplete -- written
    concurrently, or handed back short by on-access scanning -- it executes
    garbage, and garbage bytecode produces impossible values rather than a clean
    error: a method call that returns the wrong object, an operand that is a
    code object, "unknown opcode".

    Removing this package's cache and telling children not to write one means
    every worker compiles these few small modules from source, in memory, the
    same way the parent already does successfully.  Third-party caches under
    site-packages are untouched, so numpy and friends still load at full speed.
    """
    try:
        pkg = Path(__file__).resolve().parent
        for cache in pkg.rglob("__pycache__"):
            for pyc in cache.glob("*.pyc"):
                pyc.unlink(missing_ok=True)
        # inherited by spawned children
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    except Exception:  # pragma: no cover - best effort only
        pass


def _runs_serially(cfg: SimConfig) -> bool:
    """Does this configuration run in *this* process?

    Used to tell an environment fault from a real bug after a worker dies. If
    the identical configuration completes here, the simulation is not what
    failed.
    """
    try:
        run_sim(cfg)
        return True
    except Exception:
        return False


def _init_worker() -> None:
    """Keep each worker single-threaded, and read-only towards the cache.

    The numeric work here is tiny (2x2 eigenvalues, small least squares), but
    numpy's BLAS will happily start a thread per core *inside every worker*.
    With one worker per core that is N^2 threads competing for N cores, which
    is slower than doing nothing about it.

    ``dont_write_bytecode`` is the other half of :func:`_precompile`: the parent
    has already written the cache, so no worker has any reason to write to it.
    """
    sys.dont_write_bytecode = True
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


def _worker(item):
    """Run one configuration, tagged with its index so order can be restored."""
    idx, cfg = item
    return idx, run_sim(cfg)


def _say(msg: str) -> None:
    """Print immediately.

    Worth being explicit about: Python block-buffers stdout whenever it is not
    a terminal (a pipe, a redirect, a CI log), so without ``flush`` a long batch
    looks frozen and then dumps everything at the end.
    """
    print(msg, flush=True)


def run_many(
    configs: Sequence[SimConfig],
    jobs: Optional[int] = None,
    progress: bool = True,
    label: str = "",
) -> List[WindowSeries]:
    """Run configurations in parallel, preserving input order.

    Progress is reported as runs *complete*, not at the end of the batch, so a
    long sweep is visibly alive.  On Windows and macOS, ``multiprocessing``
    spawns fresh interpreters that must re-import numpy and simpy, which can
    take tens of seconds before the first result appears; the "starting" line is
    printed before that so the wait is legible.
    """
    jobs = jobs or default_jobs()
    configs = list(configs)
    n = len(configs)
    # never start more workers than there is work for them to do
    jobs = max(1, min(jobs, n))
    tag = f"[{label}] " if label else ""

    if progress:
        _say(f"  {tag}{n} runs on {jobs} worker(s), starting...")

    t0 = time.time()
    out: List[Optional[WindowSeries]] = [None] * n
    done = 0
    # report about ten times over the batch, and at least every run when small
    step = max(1, n // 10)

    def note(done: int) -> None:
        if not progress:
            return
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (n - done) / rate if rate > 0 else float("inf")
        _say(f"  {tag}{done}/{n} done, {el:.0f}s elapsed"
             + (f", ~{eta:.0f}s left" if done < n and eta < 1e6 else ""))

    if jobs <= 1:
        _init_worker()
        for i, cfg in enumerate(configs):
            out[i] = run_sim(cfg)
            done += 1
            if done % step == 0 or done == n:
                note(done)
    else:
        _prepare_worker_cache()
        try:
            with Pool(jobs, initializer=_init_worker) as pool:
                for idx, ws in pool.imap_unordered(
                    _worker, list(enumerate(configs)), chunksize=1
                ):
                    out[idx] = ws
                    done += 1
                    if done % step == 0 or done == n:
                        note(done)
        except Exception as exc:
            # A worker died.  Before blaming the simulation, check whether the
            # same work runs in this process.  If it does, the defect is in the
            # worker environment, and the honest thing is to say so, finish the
            # batch serially, and let the science proceed.
            if not _runs_serially(configs[0]):
                raise
            _say("")
            _say("  " + "!" * 66)
            _say(f"  A worker process failed: {type(exc).__name__}: {exc}")
            _say("")
            _say("  The identical configuration completes in this process, so the")
            _say("  simulation is not what broke -- a worker executed corrupt code.")
            _say("  Values that cannot exist (a code object where a float belongs,")
            _say("  'unknown opcode') mean memory or file contents were altered")
            _say("  underneath the interpreter. Usual causes, in order:")
            _say("")
            _say("    * on-access virus scanning or cloud sync on the project drive")
            _say("        -> exclude the folder, or move the project to a local disk")
            _say("    * faulty RAM, which shows up under many processes first")
            _say("        -> run Windows Memory Diagnostic (mdsched.exe) overnight")
            _say("    * too many workers for the memory available")
            _say("        -> --jobs 4 is usually enough; the runs are seconds each")
            _say("")
            _say("  Falling back to a single process for this batch. Results are")
            _say("  identical -- seeds are fixed, so parallelism changes nothing")
            _say("  but wall-clock.")
            _say("  " + "!" * 66)
            _say("")
            _init_worker()
            out = [None] * n
            done = 0
            for i, cfg in enumerate(configs):
                out[i] = run_sim(cfg)
                done += 1
                if done % step == 0 or done == n:
                    note(done)

    if progress:
        dt = time.time() - t0
        _say(f"  {tag}finished {n} runs in {dt:.1f}s ({dt / max(n, 1):.2f}s/run)")
    return [ws for ws in out if ws is not None]


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclass structures into JSON types."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    if is_dataclass(obj) and not isinstance(obj, type):
        return jsonable(vars(obj))
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    return str(obj)


def save_json(path: Path, payload: Dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True))
    print(f"  wrote {path.relative_to(ROOT) if ROOT in path.parents else path}")
    return path


def load_json(path: Path) -> Dict:
    return json.loads(Path(path).read_text())
