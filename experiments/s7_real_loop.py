"""S7 -- closed-loop real-agent test of the predicted boundary (added in revision).

S6 has no feedback: each answer is independent of what was verified before.
S7 closes the loop on the same real components (local LLM answers, a real LLM
verifier behind a shared pool with a deadline) and then runs the method of
S2 -> freeze -> S3 on it, reusing ``ecfi.identify`` and ``ecfi.detect``
unchanged.

The loop
--------
*  **Epistemic state u_k.**  Tasks complete in windows of ``W``.  ``u_k`` is the
   fraction of window k's final results that are wrong, scored against ground
   truth -- an audit signal, the same observable the simulator's policy uses.
*  **How it changes later demand.**  During the next window every arriving
   agent verifies with probability ``v = sat_[0,1](v_bar + kappa (u_k - u_ref))``,
   where ``u_ref`` is a slow EWMA of ``u`` (the operating point).  ``kappa`` is
   the gain being swept.  The one-window delay is the simulator's.
*  **Epistemic coupling.**  With probability ``P_DEP`` a task's result is the
   sum of its own product and an upstream product read from a shared store of
   the last ``W`` results (the addition is exact, done in code).  It is wrong
   if either addend is.  So an answer that went unverified in one window
   damages results in the next.
*  **Resource coupling.**  Verification is a live LLM call through a pool of
   ``C`` slots with a deadline on the wait; on timeout the request is cancelled
   and the unverified answer stands, as in S6.
*  **Congestion q_k.**  Mean, over the window's arrivals, of waiting requests
   per slot.

Direct answers are produced once into a task bank on an idle server and
replayed; verification calls, queueing, timeouts, the store and the policy are
live.  A run's final results therefore depend on the loop even though its
direct answers do not.

Phases
------
``identify``  three excitations at once -- hard-task bursts (ambiguity),
              background-load pulses, dithered nominal effort -- then OLS for
              a..f, leave-one-run-out fit, baseline stability, bootstrap CI for
              kappa_crit.  Writes the freeze, including the sweep grid.
``sweep``     independent seeds, no excitation, gain fixed per run at the frozen
              grid.  Refuses to start without the freeze.

Usage:  OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=2 ollama serve &
        python experiments/s7_real_loop.py identify --ollama http://127.0.0.1:11435/api/chat
        python experiments/s7_real_loop.py sweep    --ollama http://127.0.0.1:11435/api/chat
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import s6_real_agents as s6  # noqa: E402
from ecfi.detect import margin_knee, variance_extrapolation  # noqa: E402
from ecfi.identify import build_design, identify  # noqa: E402
from ecfi.linear import LocalParams, is_schur, stability_margin  # noqa: E402
from ecfi.policies import make_policy  # noqa: E402
from ecfi.runner import FROZEN, RESULTS, load_json, save_json  # noqa: E402
from ecfi.sim import WindowSeries  # noqa: E402
from s2_identify import git_commit  # noqa: E402
from s3_sweep import metrics  # noqa: E402

# --- pre-registered in frozen/prereg.md, addendum 3 -------------------------
W = 40                      # tasks per window
WARMUP = 15                 # windows discarded
RATE = 2.5                  # task arrivals per second
C = 2                       # verifier slots (needs a server with >= 2 parallel)
DEADLINE = 1.5              # seconds waiting for a slot
V_BAR = 0.4                 # nominal verification probability
P_DEP = 0.5                 # P(task consumes an upstream result)
REF_TAU = 20.0              # set-point time constant, windows
BG_HOLD = 1.0               # mean seconds a background tenant holds a slot

KAPPA_BASE = 1.0            # identification operating gain
IDENT_SEEDS = (101, 102, 103)
IDENT_WINDOWS = 135
V_DITHER, V_BLOCK = 0.25, 3             # v_bar +/- 0.25, redrawn every 3 windows
BG_LEVEL, BG_PERIOD, BG_ON = 0.5, 11, 5  # pulse: utilisation 0.5 for 5 of 11 windows
HARD_LEVEL, HARD_PERIOD, HARD_ON = 0.6, 7, 3

SWEEP_SEEDS = (1, 2, 3)
SWEEP_WINDOWS = 95
GRID_MULT = (0.0, 0.5, 0.8, 1.0, 1.25, 1.6)   # x kappa_hat_crit
GRID_FALLBACK = (0.0, 1.0, 2.0, 3.0, 4.5, 6.0)  # if the freeze is unusable

BANK = {"easy": 2000, "hard": 600}


def task_bank(model: str, path: Path) -> list:
    """Direct answers, produced once on an idle server and replayed."""
    if path.exists():
        return load_json(path)["tasks"]
    rng, bank = random.Random(0), []
    for hard, n in ((False, BANK["easy"]), (True, BANK["hard"])):
        for _ in range(n):
            a = rng.randint(1001, 9999) if hard else rng.randint(101, 999)
            b = rng.randint(11, 99)
            bank.append({"a": a, "b": b, "hard": hard, "primary": s6.primary(model, a, b)[0]})
    save_json(path, {"model": model, "tasks": bank})
    return bank


def run_loop(bank, seed, kappa, n_windows, model, vbar=lambda k: V_BAR,
             bg=lambda k: 0.0, hard=lambda k: 0.0, rate=RATE) -> WindowSeries:
    """One closed-loop run.  Schedules are functions of the window index."""
    rng = random.Random(seed)
    easy_b = [t for t in bank if not t["hard"]]
    hard_b = [t for t in bank if t["hard"]]
    pool, stop, lock = s6.VerifierPool(C), threading.Event(), threading.Lock()
    store: deque = deque(maxlen=W)          # own-result correctness of recent tasks
    windows, cur, err = [], [], []
    sig = {"u": 0.0, "ref": 0.0}            # equal until the first window closes: v = v_bar
    t_start = time.monotonic()

    def close_window():
        col = lambda j: np.array([r[j] for r in cur], dtype=float)  # noqa: E731
        u = float(col(0).mean())
        windows.append({
            "u": u, "q": float(col(1).mean()), "v": float(col(2).mean()),
            "wait": float(col(3).mean()), "wait_p95": float(np.percentile(col(3), 95)),
            "fallback": float(col(4).mean()), "amb": float(col(5).mean()),
            "bg": float(col(6).mean()), "tokens": float(col(7).sum()),
            "kappa": kappa, "t_end": time.monotonic() - t_start,
        })
        sig["ref"] = u if len(windows) == 1 else sig["ref"] + (u - sig["ref"]) / REF_TAU
        sig["u"] = u
        cur.clear()

    def agent(task, d_verify, d_dep, d_parent):
        try:
            a, b = task["a"], task["b"]
            with lock:
                k, u_sig, u_ref = len(windows), sig["u"], sig["ref"]
                parent_wrong = (store[int(d_parent * len(store))]
                                if store and d_dep < P_DEP else False)
            q = pool.q_obs()
            v = make_policy(kappa, 0.0, vbar(k), u_ref, 1.0).effort(u_sig, q)
            own, wait, timed_out, served = task["primary"], 0.0, False, 0
            if d_verify < v:
                out, wait, timed_out = pool.request(lambda: s6.verify(model, a, b), DEADLINE)
                if not timed_out:
                    served = 1
                    if out[0] is not None:
                        own = out[0]
            own_wrong = own != a * b
            with lock:
                store.append(own_wrong)
                # result = own product (+ upstream product): wrong if either is
                cur.append((own_wrong or parent_wrong, q, v, wait, timed_out,
                            task["hard"], bg(k), served))
                if len(cur) >= W:
                    close_window()
        except Exception as exc:  # surfaced by the arrival loop
            err.append(exc)

    def tenants():
        """Background traffic: occupy a slot by sleeping; no model calls."""
        r = random.Random(seed + 1)
        while not stop.is_set():
            level = bg(len(windows))
            if level <= 0.0:
                stop.wait(0.25)
            elif not stop.wait(r.expovariate(level * C / BG_HOLD)):
                pool.submit(lambda h=r.expovariate(1.0 / BG_HOLD): time.sleep(h))

    threading.Thread(target=tenants, daemon=True).start()
    ex, i = ThreadPoolExecutor(max_workers=64), 0
    try:
        while len(windows) < n_windows:
            if err:
                raise err[0]
            time.sleep(rng.expovariate(rate))
            src = hard_b if rng.random() < hard(i // W) else easy_b
            ex.submit(agent, rng.choice(src), rng.random(), rng.random(), rng.random())
            i += 1
    finally:
        stop.set()
        ex.shutdown(wait=True)
        pool.drain()

    w = windows[WARMUP:n_windows]
    get = lambda key: np.array([x[key] for x in w], dtype=float)  # noqa: E731
    return WindowSeries(**{key: get(key) for key in w[0]},
                        meta={"seed": seed, "arrival_rate": i / (time.monotonic() - t_start)})


def as_series(d: dict) -> WindowSeries:
    return WindowSeries(**{k: np.asarray(v, dtype=float) for k, v in d.items() if k != "meta"})


def held_out_r2(series: list) -> dict:
    """Leave-one-run-out R^2 of both equations."""
    out = {"u": [], "q": []}
    for j, test in enumerate(series):
        fit = identify(series[:j] + series[j + 1:])
        Xu, yu, Xq, yq = build_design([test])
        for key, X, y, reg in (("u", Xu, yu, fit.reg_u), ("q", Xq, yq, fit.reg_q)):
            out[key].append(1.0 - float(((y - X @ reg.coef) ** 2).sum()) / float((y ** 2).sum()))
    return out


def phase_identify(args, bank) -> int:
    kc_path = args.frozen / "kcrit.json"
    if kc_path.exists() and not args.refreeze:
        print(f"refusing to overwrite {kc_path}: the freeze is the pre-registration.")
        return 1
    res_path = args.results / "s7_identify.json"
    saved = load_json(res_path) if res_path.exists() else {"runs": {}}
    for seed in args.ident_seeds:
        if str(seed) in saved["runs"]:
            continue
        r = random.Random(seed)
        blocks = [V_BAR + r.uniform(-V_DITHER, V_DITHER)
                  for _ in range(args.ident_windows // V_BLOCK + 2)]
        ws = run_loop(
            bank, seed, KAPPA_BASE, args.ident_windows, args.model,
            vbar=lambda k: blocks[min(k // V_BLOCK, len(blocks) - 1)],
            bg=lambda k: BG_LEVEL if (k - 3) % BG_PERIOD < BG_ON else 0.0,
            hard=lambda k: HARD_LEVEL if (k - 5) % HARD_PERIOD < HARD_ON else 0.0)
        saved["runs"][str(seed)] = {**ws.as_dict(), "meta": ws.meta}
        save_json(res_path, saved)
        print(f"  identification run seed {seed}: u {ws.u.mean():.3f}  v {ws.v.mean():.3f}  "
              f"q {ws.q.mean():.3f}  fallback {ws.fallback.mean():.3f}", flush=True)

    series = [as_series(saved["runs"][str(s)]) for s in args.ident_seeds]
    res = identify(series, bootstrap=600, rng=np.random.default_rng(7))
    print("\n" + res.summary())
    p, kc = res.params, res.kappa_crit
    ho = held_out_r2(series) if len(series) > 1 else {"u": [], "q": []}
    gate = {
        "r2_u": res.reg_u.r2 > 0.10, "r2_q": res.reg_q.r2 > 0.10,
        "corr_uv": abs(res.excitation["corr_uv"]) < 0.95,
        "held_out_u": bool(ho["u"]) and min(ho["u"]) > 0.0,
        "held_out_q": bool(ho["q"]) and min(ho["q"]) > 0.0,
        "baseline_stable": is_schur(p, 0.0),
        "signs_bcef": p.b > 0 and p.c > 0 and p.e > 0 and p.f > 0,
        "kappa_crit_usable": bool(np.isfinite(kc) and kc > 0),
    }
    usable = gate["kappa_crit_usable"]
    grid = [m * kc for m in GRID_MULT] if usable else list(GRID_FALLBACK)
    print(f"\n  held-out R2  u: {np.round(ho['u'], 3).tolist()}   q: {np.round(ho['q'], 3).tolist()}")
    print("  gate: " + "  ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in gate.items()))
    if not all(gate.values()):
        print("  WARNING: gate not fully met -- reported as a scope limit, not as support.")
    print(f"  kappa_hat_crit = {kc:.3f}  CI {res.kappa_crit_ci}\n  sweep grid: {np.round(grid, 3).tolist()}")

    stamp = {"frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
             "model": args.model, "W": W, "rate": RATE, "C": C, "deadline": DEADLINE,
             "v_bar": V_BAR, "p_dep": P_DEP, "kappa_base": KAPPA_BASE,
             "seeds": list(args.ident_seeds), "n_windows": args.ident_windows}
    save_json(args.frozen / "params.json", {**stamp, **res.to_dict(), "held_out_r2": ho, "gate": gate})
    save_json(kc_path, {**stamp, "kappa_crit": kc, "params": p.as_dict(),
                        "kappa_crit_ci": list(res.kappa_crit_ci) if res.kappa_crit_ci else None,
                        "gate": gate, "gate_passed": all(gate.values()),
                        "grid": grid, "grid_is_fallback": not usable,
                        "note": "frozen before the S7 sweep; do not regenerate afterwards"})
    return 0


def aggregate(frozen: dict, runs: dict) -> dict:
    grid = frozen["grid"]
    by_k = {k: [as_series(r) for r in runs.values() if r["meta"]["kappa"] == k] for k in grid}
    base_var = float(np.mean([np.var(ws.u) for ws in by_k[grid[0]]])) or 1e-12
    p = LocalParams(**frozen["params"])
    out = {"per_kappa": {}, "crossing_kappa": None}
    for k in grid:
        rows = [metrics(ws, base_var) for ws in by_k[k]]
        n = len(rows)
        agg = {key: [float(np.mean([r[key] for r in rows])),
                     float(np.std([r[key] for r in rows], ddof=1) / np.sqrt(n)) if n > 1 else 0.0]
               for key in rows[0]}
        agg["predicted_margin"] = stability_margin(p, k)
        # pre-registered crossing criterion
        acf_hi = agg["acf1_u"][0] + (stats.t.ppf(0.975, n - 1) * agg["acf1_u"][1] if n > 1 else np.inf)
        agg["crossed"] = bool(k > 0 and acf_hi < 0.0 and agg["var_ratio_u"][0] >= 2.0)
        if agg["crossed"] and out["crossing_kappa"] is None:
            out["crossing_kappa"] = k
        out["per_kappa"][str(k)] = agg
    pk = out["per_kappa"]
    out["kappa_star_margin_knee"] = margin_knee(grid, [pk[str(k)]["margin"][0] for k in grid])
    out["kappa_star_var_extrapolation"] = variance_extrapolation(
        grid, [pk[str(k)]["var_ratio_u"][0] for k in grid])
    return out


def phase_sweep(args, bank) -> int:
    kc_path = args.frozen / "kcrit.json"
    if not kc_path.exists():
        print(f"{kc_path} is missing: run the identify phase first.")
        return 1
    frozen = load_json(kc_path)
    grid, k_hat = frozen["grid"], frozen["kappa_crit"]
    print(f"frozen kappa_hat_crit = {k_hat:.3f}  CI {frozen['kappa_crit_ci']}  "
          f"(frozen at {frozen['frozen_at']}, gate passed: {frozen['gate_passed']})")
    res_path = args.results / "s7_sweep.json"
    saved = load_json(res_path) if res_path.exists() else {"frozen": frozen, "runs": {}}
    todo = [(k, s) for k in grid for s in args.sweep_seeds]
    random.Random(0).shuffle(todo)          # run order randomised over gain and seed
    for n, (k, seed) in enumerate(todo, 1):
        key = f"{k:.4f}/{seed}"
        if key in saved["runs"]:
            continue
        ws = run_loop(bank, 1000 * seed + grid.index(k), k, args.sweep_windows, args.model)
        saved["runs"][key] = {**ws.as_dict(), "meta": {**ws.meta, "kappa": k, "order": n}}
        save_json(res_path, saved)
        print(f"  [{n}/{len(todo)}] kappa {k:.3f} seed {seed}: u {ws.u.mean():.3f}  "
              f"sd(u) {ws.u.std():.3f}  v {ws.v.mean():.3f}  fallback {ws.fallback.mean():.3f}",
              flush=True)

    a = saved["summary"] = aggregate(frozen, saved["runs"])
    save_json(res_path, saved)
    print("\n   kappa   k/khat   error     v      q     fb   acf1(u)  varratio  nyq(v)  "
          "margin  predicted  crossed")
    for k in grid:
        m = a["per_kappa"][str(k)]
        print(f"  {k:6.2f}   {k / k_hat if k_hat else float('nan'):5.2f}   {m['u'][0]:.3f}  "
              f"{m['v'][0]:.3f}  {m['q'][0]:.3f}  {m['fallback'][0]:.3f}  "
              f"{m['acf1_u'][0]:+.3f}   {m['var_ratio_u'][0]:6.2f}   {m['nyq_v'][0]:.3f}  "
              f"{m['margin'][0]:+.3f}    {m['predicted_margin']:+.3f}    {m['crossed']}")
    for name, key in (("margin knee", "kappa_star_margin_knee"),
                      ("variance extrapolation", "kappa_star_var_extrapolation")):
        ks = a[key]
        print(f"   observed {name:23s} = {ks:.3f}   ratio to prediction {ks / k_hat:.2f}"
              if np.isfinite(ks) and k_hat else f"   observed {name:23s} = n/a")
    print(f"   crossing criterion first met at kappa = {a['crossing_kappa']}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("identify", "sweep"))
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--ollama", default=s6.OLLAMA, help="chat endpoint; needs >= C parallel slots")
    ap.add_argument("--refreeze", action="store_true", help="overwrite the freeze")
    # the remaining flags exist for smoke tests; the study uses the defaults
    ap.add_argument("--frozen", type=Path, default=FROZEN / "real_agents")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--ident-seeds", type=int, nargs="+", default=list(IDENT_SEEDS))
    ap.add_argument("--sweep-seeds", type=int, nargs="+", default=list(SWEEP_SEEDS))
    ap.add_argument("--ident-windows", type=int, default=IDENT_WINDOWS)
    ap.add_argument("--sweep-windows", type=int, default=SWEEP_WINDOWS)
    args = ap.parse_args()

    s6.OLLAMA = args.ollama
    s6.chat(args.model, "hi", 1)
    bank = task_bank(args.model, RESULTS / "s7_task_bank.json")
    return (phase_identify if args.phase == "identify" else phase_sweep)(args, bank)


if __name__ == "__main__":
    raise SystemExit(main())
