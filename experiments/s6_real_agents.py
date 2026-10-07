"""S6 -- small real-agent experiment (Section III, added in revision).

Everything in S1-S5 runs on the SimPy testbed.  This script runs S1's causal
question on *real* LLM calls: does contention for a shared verification
service damage objective correctness, and does that depend on the fallback?

It is a **correction-service experiment**.  Concurrent task agents each hold
an answer produced by a real model and compete for a real LLM verifier with
limited capacity, against synthetic competing traffic.

*  **Answering.**  One direct call per task: "What is a * b? Reply with only
   the number."  Uncertainty is ``u_hat = 1 - exp(sum of answer-token
   logprobs)``.  Answers are produced **once per seed, before any arm, on an
   idle server**, and the identical saved answers are replayed in every arm.
   Answering is therefore isolated from verification contention by
   construction, and arms differ only in which answers got verified.
*  **Agents.**  Each arriving task is one agent: it reads its saved answer and
   the queue, the policy decides whether to verify, and it then waits for the
   shared verifier or falls back.  Arrivals are Poisson; observed concurrency
   is reported per arm, not assumed.
*  **Verifier.**  A second call to the same model that recomputes the product
   from two partial products, without seeing the agent's answer.  Its answer
   replaces the agent's; if it returns no parseable answer the agent's stands.
   It is not a tool and has no access to ground truth, so it can be wrong.
*  **Shared service.**  ``VerifierPool``: C slots, FIFO.  Ollama serves one
   request at a time per model, so C = 1 is a real inference-server slot.
*  **Background traffic.**  Synthetic tenants (Poisson arrivals, exponential
   hold) that occupy a verifier slot by sleeping.  They make no model calls.
   Realised occupancy is measured and reported.
*  **Policies** (``ecfi.policies``, effort read as verification probability,
   ``v_max = 1``): ``always`` (v = 1), ``proportional`` (v = kappa * u_hat),
   ``aware`` (v = kappa * u_hat / (1 + gamma * q)).
*  **Fallback.**  ``wait``: no deadline, every requested verification runs.
   ``skip``: a request that has not *started service* within the deadline is
   cancelled and removed from the queue -- it never occupies capacity -- and
   the agent's unverified answer is returned.  A verification that has started
   is never cut off.

Scoring is exact integer match against ``a * b`` computed here; a reply with
no number is an error.  Arm order is randomised per seed and the pool is
drained between arms.  There is no shared context store, so this measures the
*direct* fallback term of ``b`` only, and makes no claim about instability.

Usage:  ollama pull qwen2.5:3b
        python experiments/s6_real_agents.py [--seeds N] [--tasks N] [--model NAME]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecfi.policies import make_policy  # noqa: E402
from ecfi.runner import RESULTS, save_json  # noqa: E402

OLLAMA = "http://localhost:11434/api/chat"

ARMS = tuple(f"{p}/{f}/{lvl}"
             for p, f in (("always", "wait"), ("always", "skip"),
                          ("proportional", "skip"), ("aware", "skip"))
             for lvl in ("low", "high"))


def chat(model: str, prompt: str, num_predict: int, logprobs: bool = False) -> dict:
    body = {"model": model, "stream": False, "think": False, "logprobs": logprobs,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": 0, "num_predict": num_predict}}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def last_int(text: str, pattern: str = r"(\d+)"):
    m = re.findall(pattern, text.replace(",", ""))
    return int(m[-1]) if m else None


def primary(model: str, a: int, b: int):
    """Direct answer and logprob-based uncertainty."""
    d = chat(model, f"What is {a} * {b}? Reply with only the number.", 16, logprobs=True)
    conf = math.exp(sum(t["logprob"] for t in d.get("logprobs") or []))
    return last_int(d["message"]["content"]), 1.0 - conf


def verify(model: str, a: int, b: int):
    """Recomputation from partial products; returns (answer or None, service time)."""
    t0 = time.monotonic()
    tens, ones = b // 10 * 10, b % 10
    d = chat(model, f"Compute {a} * {b}. Output exactly three lines and nothing else:\n"
                    f"{a} * {tens} = <number>\n{a} * {ones} = <number>\n"
                    "ANSWER: <sum of the two numbers>", 80)
    return last_int(d["message"]["content"], r"ANSWER:\D*(\d+)"), time.monotonic() - t0


class VerifierPool:
    """The shared service: C slots, FIFO, optional deadline on the wait."""

    def __init__(self, c: int):
        self.c = c
        self._ex = ThreadPoolExecutor(max_workers=c)
        self._lock = threading.Lock()
        self.waiting = 0

    def q_obs(self) -> float:
        """Waiting requests per slot -- same observable as ``Testbed.queue_obs``."""
        return self.waiting / self.c

    def _bump(self, d: int) -> None:
        with self._lock:
            self.waiting += d

    def submit(self, fn):
        started, box, t0 = threading.Event(), {}, time.monotonic()

        def job():
            self._bump(-1)
            box["wait"] = time.monotonic() - t0
            started.set()
            return fn()

        self._bump(+1)
        return self._ex.submit(job), started, box, t0

    def request(self, fn, deadline=None):
        """Run ``fn`` in a slot.  Returns (result, wait, timed_out).

        The deadline is on *waiting for a slot*, as in the simulator.  A
        request that times out is cancelled before it starts, so it never
        occupies capacity; one that has started service is never abandoned.
        """
        fut, started, box, t0 = self.submit(fn)
        if not started.wait(deadline) and fut.cancel():
            self._bump(-1)
            return None, time.monotonic() - t0, True
        return fut.result(), box["wait"], False

    def drain(self) -> None:
        """Cancel everything queued and wait for the slot(s) to go idle."""
        self._ex.shutdown(wait=True, cancel_futures=True)


def background(pool: VerifierPool, rate: float, hold: float, stop: threading.Event,
               rng: random.Random, held: list) -> None:
    """Exogenous tenants: occupy a slot, carry no epistemics, use no GPU."""
    def tenant(h):
        time.sleep(h)
        held.append(h)

    while not stop.wait(rng.expovariate(rate)):
        pool.submit(lambda h=rng.expovariate(1.0 / hold): tenant(h))


def agent(i, task, answer, policy, pool, deadline, draw, model, t_arm) -> dict:
    (a, b), (ans, u_hat) = task, answer
    t0 = time.monotonic()
    q = pool.q_obs()
    v = policy.effort(u_hat, q)
    rec = {"i": i, "a": a, "b": b, "primary": ans, "u_hat": u_hat, "q_obs": q, "v": v,
           "requested": draw < v, "verified": False, "timeout": False,
           "wait": 0.0, "service": 0.0}
    final = ans
    if rec["requested"]:
        out, wait, timed_out = pool.request(lambda: verify(model, a, b), deadline)
        rec.update(wait=wait, timeout=timed_out)
        if not timed_out:
            v_ans, service = out
            rec.update(verified=True, service=service, verifier_answer=v_ans)
            if v_ans is not None:
                final = v_ans
    t1 = time.monotonic()
    rec.update(final=final, wrong=final != a * b, primary_wrong=ans != a * b,
               t_arrive=t0 - t_arm, latency=t1 - t0)
    return rec


def run_arm(name, tasks, answers, gaps, draws, args, seed) -> dict:
    policy_name, fallback, level = name.split("/")
    policy = {
        "always": make_policy(0.0, v_bar=1.0, v_max=1.0),
        "proportional": make_policy(args.kappa, v_bar=0.0, v_max=1.0),
        "aware": make_policy(args.kappa, gamma=args.gamma, v_bar=0.0, v_max=1.0),
    }[policy_name]
    deadline = args.deadline if fallback == "skip" else None
    pool, stop, held = VerifierPool(args.verifiers), threading.Event(), []
    t_arm = time.monotonic()
    if level == "high":
        rate = args.bg_load * args.verifiers / args.bg_hold
        threading.Thread(target=background, daemon=True,
                         args=(pool, rate, args.bg_hold, stop, random.Random(seed), held)).start()
    try:
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            futs = []
            for i, task in enumerate(tasks):
                time.sleep(gaps[i])
                futs.append(ex.submit(agent, i, task, answers[i], policy, pool, deadline,
                                      draws[i], args.model, t_arm))
            recs = [f.result() for f in futs]
        duration = time.monotonic() - t_arm
    finally:
        stop.set()
        pool.drain()
    return {"summary": summarize(recs, duration, sum(held), args.verifiers), "tasks": recs}


def summarize(recs: list, duration: float, bg_held: float, c: int) -> dict:
    req = [r for r in recs if r["requested"]]
    done = [r for r in req if r["verified"]]
    col = lambda rows, k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    mean = lambda x: float(x.mean()) if len(x) else 0.0               # noqa: E731
    p95 = lambda x: float(np.percentile(x, 95)) if len(x) else 0.0    # noqa: E731
    # agents in the system (waiting for or in verification): +1 at arrival, -1 at exit
    events = sorted([(r["t_arrive"], 1) for r in recs]
                    + [(r["t_arrive"] + r["latency"], -1) for r in recs])
    in_flight = np.cumsum([e[1] for e in events])
    return {
        "n": len(recs),
        "error": mean(col(recs, "wrong")),
        "primary_error": mean(col(recs, "primary_wrong")),
        "verify_requested": len(req),
        "verify_completed": len(done),          # verifications, not tasks: every task completes
        "timeouts": len(req) - len(done),
        # abandoned requests enter at the time they gave up (censored at D)
        "wait_mean": mean(col(req, "wait")), "wait_p95": p95(col(req, "wait")),
        "q_mean": mean(col(recs, "q_obs")),
        "service_mean": mean(col(done, "service")),
        # time from arrival to final answer; excludes the precomputed answering call
        "latency_mean": mean(col(recs, "latency")), "latency_p95": p95(col(recs, "latency")),
        "fixed": sum(r["primary_wrong"] and not r["wrong"] for r in recs),
        "broken": sum(r["wrong"] and not r["primary_wrong"] for r in recs),
        "verifier_no_answer": sum(r.get("verifier_answer", 0) is None for r in done),
        "agents_in_flight_mean": float(col(recs, "latency").sum() / duration),
        "agents_in_flight_max": int(in_flight.max()),
        "arrival_rate": len(recs) / duration,
        "agent_occupancy": float(col(done, "service").sum() / (duration * c)),
        "bg_occupancy": bg_held / (duration * c),
        "duration": duration,
    }


def paired_contrast(arms: dict) -> dict:
    """S1 analogue: does load raise error more under ``skip`` than under ``wait``?

    Per-task difference-in-differences over the paired tasks, always-verify.
    The se describes task variation within one run; replication across runs is
    what ``across_seeds`` reports.
    """
    w = lambda name: np.array([r["wrong"] for r in arms[name]["tasks"]], dtype=float)  # noqa: E731
    d_skip = w("always/skip/high") - w("always/skip/low")
    d_wait = w("always/wait/high") - w("always/wait/low")
    dd = d_skip - d_wait
    return {"delta_error_skip": float(d_skip.mean()), "delta_error_wait": float(d_wait.mean()),
            "diff_in_diff": float(dd.mean()),
            "diff_in_diff_se": float(dd.std(ddof=1) / np.sqrt(len(dd)))}


def across_seeds(by_seed: dict) -> dict:
    """The seed is the unit of replication, as in S1."""
    runs = list(by_seed.values())
    n = len(runs)

    def agg(x):
        x = np.asarray(x, dtype=float)
        sd = float(x.std(ddof=1)) if n > 1 else 0.0
        half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else float("nan")
        return {"mean": float(x.mean()), "sd": sd, "ci95": [float(x.mean()) - half, float(x.mean()) + half]}

    keys = ("error", "primary_error", "verify_requested", "verify_completed", "timeouts",
            "wait_mean", "wait_p95", "q_mean", "service_mean", "latency_p95",
            "agents_in_flight_mean", "agents_in_flight_max", "agent_occupancy", "bg_occupancy")
    out = {"arms": {a: {k: agg([r["arms"][a]["summary"][k] for r in runs]) for k in keys}
                    for a in ARMS}}
    dd = [r["contrast"]["diff_in_diff"] for r in runs]
    out["diff_in_diff"] = {"per_seed": dd, **agg(dd)}
    if n > 1:
        out["diff_in_diff"]["p_two_sided"] = float(stats.ttest_1samp(dd, 0.0).pvalue)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:3b")
    ap.add_argument("--seeds", type=int, default=5, help="independent runs, seeds 1..N")
    ap.add_argument("--tasks", type=int, default=100, help="tasks per arm")
    ap.add_argument("--interarrival", type=float, default=1.6, help="mean seconds between tasks")
    ap.add_argument("--verifiers", type=int, default=1, help="C; Ollama serves 1 at a time")
    ap.add_argument("--deadline", type=float, default=1.5, help="D, seconds waiting for a slot")
    ap.add_argument("--bg-load", type=float, default=0.4, help="background utilisation, high level")
    ap.add_argument("--bg-hold", type=float, default=1.0, help="mean seconds a tenant holds a slot")
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--out", default=str(RESULTS / "s6_real_agents.json"))
    args = ap.parse_args()

    chat(args.model, "hi", 1)  # load the model before the clock starts
    payload = {"config": vars(args), "seeds": {}}
    for seed in range(1, args.seeds + 1):
        rng = random.Random(seed)
        tasks = [(rng.randint(101, 999), rng.randint(11, 99)) for _ in range(args.tasks)]
        gaps = [rng.expovariate(1.0 / args.interarrival) for _ in tasks]
        draws = [rng.random() for _ in tasks]
        order = rng.sample(ARMS, len(ARMS))

        # answered once, serially, on an idle server; replayed in every arm
        answers = [primary(args.model, a, b) for a, b in tasks]

        run = payload["seeds"][seed] = {"order": order, "arms": {}}
        print(f"\nseed {seed}  (arms in run order)")
        print("  arm                        error  primary  req  done  t/out  "
              "wait   p95w   q     svc  p95lat  fixed broken  in-flight  bg-occ")
        for name in order:
            arm = run["arms"][name] = run_arm(name, tasks, answers, gaps, draws, args, seed)
            s = arm["summary"]
            print(f"  {name:25s}  {s['error']:.3f}  {s['primary_error']:.3f}   "
                  f"{s['verify_requested']:3d}  {s['verify_completed']:3d}   {s['timeouts']:3d}  "
                  f"{s['wait_mean']:5.2f}  {s['wait_p95']:5.2f}  {s['q_mean']:4.2f}  "
                  f"{s['service_mean']:4.2f}  {s['latency_p95']:5.2f}  "
                  f"{s['fixed']:4d}  {s['broken']:4d}   "
                  f"{s['agents_in_flight_mean']:4.2f}/{s['agents_in_flight_max']:<2d}   "
                  f"{s['bg_occupancy']:.2f}", flush=True)
            time.sleep(2.0)  # pool is already drained; let the server settle
        c = run["contrast"] = paired_contrast(run["arms"])
        print(f"  always-verify, error(high) - error(low):  skip {c['delta_error_skip']:+.3f}"
              f"   wait {c['delta_error_wait']:+.3f}   DiD {c['diff_in_diff']:+.3f}"
              f" +/- {c['diff_in_diff_se']:.3f} (within-run se)")
        payload["across_seeds"] = across_seeds(payload["seeds"])
        save_json(Path(args.out), payload)  # after every seed: a full run is hours

    a = payload["across_seeds"]
    print(f"\nacross {args.seeds} seeds, mean (sd)")
    print("  arm                        error          req    t/out   wait   in-flight")
    for name in ARMS:
        m = a["arms"][name]
        print(f"  {name:25s}  {m['error']['mean']:.3f} ({m['error']['sd']:.3f})  "
              f"{m['verify_requested']['mean']:5.1f}  {m['timeouts']['mean']:5.1f}  "
              f"{m['wait_mean']['mean']:5.2f}   {m['agents_in_flight_mean']['mean']:.2f}")
    d = a["diff_in_diff"]
    print(f"\n  DiD per seed: {[round(x, 3) for x in d['per_seed']]}")
    if args.seeds > 1:
        print(f"  mean {d['mean']:+.3f}, 95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"(t, {args.seeds - 1} df), p = {d['p_two_sided']:.4f}")


if __name__ == "__main__":
    main()
