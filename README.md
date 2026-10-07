# ECFI — Epistemic–Compute Feedback Instability

Reference implementation for *"Epistemic–Compute Feedback Instability in
Networked AI Agents: When Local Verification Destabilizes the Network."*

It contains three things: the analytical core (Theorem 1 and the
verification–congestion paradox), a discrete-event testbed whose couplings are
switchable and whose ground truth is known, and the five-stage benchmark
(S1–S5) run end to end on that testbed.

The point of the testbed is **not** to show that ECFI happens. It is to test
whether a `κ̂_crit` computed from *independently identified* local parameters
predicts the transition of a *nonlinear* event-driven system **out of sample**.
If the methodology fails here, where every mechanism is known and controllable,
it cannot be trusted on live infrastructure.

---

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
python3 -m pytest -q        # 32 tests, ~15 s
```

Python ≥ 3.10. Dependencies: numpy, scipy, simpy, matplotlib, sympy.

**There is no GPU path and none is needed.** A discrete-event simulation is a
single-threaded event loop; it is CPU-bound and memory-light. What buys
wall-clock is **cores**, because replications and sweep cells are independent
and are farmed out with `multiprocessing`. Every script takes `--jobs N`.

**The default is every CPU this process is allowed to use.** That is read from
`sched_getaffinity`, not `os.cpu_count()`, so a container, cgroup, `taskset` or
Slurm allocation is respected — a job pinned to 4 CPUs on a 96-core node starts
4 workers, not 96. Each worker is also pinned to a single BLAS thread, because
one thread per core *inside* every worker would be N² threads fighting over N
cores. Two cases where you may still want to pass `--jobs` explicitly:

* **SMT / hyper-threading.** The default counts *logical* CPUs. For pure-Python
  event loops the sibling threads add roughly 10–20%, not 2×, so
  `--jobs <physical cores>` gives up very little and leaves the machine usable.
* **Sharing the machine.** Nothing here yields; it will use what you give it.

Measured: the full `run_all.sh` at 20 seeds takes about **1 hour on 2 cores**
and about **10 minutes on 16 cores**. The S3 sweeps are most of it.

## If a parallel run fails and a serial one does not

Run `python experiments/diagnose.py` first. It prints which `ecfi` is imported
from where, checksums the modules, and then runs the simulation serially, on two
workers, and at full width — so a failure is immediately attributable.

Under `spawn` (Windows, macOS) each worker is a fresh interpreter that imports
the package from scratch. A cold bytecode cache plus many simultaneous workers
can leave one of them reading a `.pyc` that another is still writing, and
executing a partially written `.pyc` produces nonsense: `SystemError: unknown
opcode`, or an `AttributeError` on a value that could not possibly hold it. The
package now compiles its own cache once, in the parent, before any worker
starts, and workers are set read-only towards it. If it still happens:

```bash
# rule out the cache entirely
PYTHONDONTWRITEBYTECODE=1 python experiments/s1_gonogo.py --jobs 8
#   PowerShell:  $env:PYTHONDONTWRITEBYTECODE=1

# confirm the simulation is healthy
python experiments/s1_gonogo.py --jobs 1 --seeds 2
```

and check that on-access virus scanning or cloud sync is not watching the
project folder — both can hand a reader a short read of a file another process
is writing.

## Run everything

```bash
JOBS=16 SEEDS=20 ./run_all.sh
```

Results land in `results/*.json`, figures in `figs/*.pdf`, and the frozen
predictions in `frozen/<regime>/`. Both ship empty: a freeze is a record of
*your* prediction and must be produced on the machine that runs the study.
Reference output from the development run is in `examples/demo-run-8seeds/`,
outside every path a script reads.

---

## The order is the method

```
S1  go/no-go       is there a causal congestion → epistemic pathway (b > 0)?
S2  identification estimate a…f, compute κ̂_crit, FREEZE it
S3  sweep          only now vary κ; compare the observed transition to the freeze
S4  strategic      measure epistemic load amplification near the boundary
S5  mitigation     congestion-aware controller on the quality/latency/cost/stability frontier
```

**S3 must not be run before S2 has frozen its prediction**, and S2 must not be
re-run afterwards. `s2_identify.py` refuses to overwrite an existing freeze
without `--refreeze`, and `s3_sweep.py` refuses to start without one. The
freeze records a UTC timestamp and the git commit; commit and tag
`frozen/` before sweeping so the ordering is verifiable by someone else.

---

## Layout

```
examples/demo-run-8seeds/   reference output; no script reads this

ecfi/
├── linear.py      A(κ), Jury conditions, κ_crit, κ_δ, u*(κ), ρ, stability margin
├── policies.py    v = κu  and  v = κu/(1+γq), with saturation restored
├── schedules.py   piecewise-constant exogenous excitation (pulses, dither)
├── sim.py         SimPy testbed: agents, verifier pool, shared context store
├── identify.py    OLS + moving-block bootstrap + excitation diagnostics
├── detect.py      transition endpoints and H4 early-warning signals
├── ela.py         normalised epistemic load amplification 𝒜_E
└── runner.py      parallel execution and result I/O

experiments/
├── regimes.py     the two operating regimes swept end to end
├── calibrate.py   model construction only — never used to fit κ_crit
├── s1_gonogo.py … s5_mitigation.py
├── analyze_s3.py  re-derive endpoints from saved sweeps without re-running
├── s6_real_agents.py  real LLM agents + a capacity-limited verifier (needs Ollama)
├── s7_real_loop.py    closed loop on the same real components: identify, freeze, sweep
└── make_figures.py
```

## How the testbed closes the loop

`N` agents process Poisson-arriving tasks carrying objective ground truth.
Verification requests contend for `C` verifier servers (lognormal service,
FIFO). Two couplings close the loop:

**Epistemic coupling (the b-mechanism).** Wrong answers are written to a shared
context store. Each task reads `n_dep` artifacts from it with probability
`p_dep`; its error probability rises with the erroneous fraction among them.
Disabling the store sets `b ≈ 0` *by construction* — the internal negative
control.

**Resource coupling.** Each task carries a verification deadline `D`. In
`RESOURCE_COUPLED` mode an expired deadline triggers skipped verification plus a
weaker model with error multiplier `m`. In `WAIT` mode deadlines are off, so
pure queueing raises latency without touching correctness.

Verification efficacy saturates: `q_fix(v) = σ(β(v − v_half))`. A cheap check
rarely catches an error; a thorough one usually does. This is the nonlinearity
the local model of Section II approximates, and `β` is the knob that decides
whether over-correction can outrun congestion — hence the two regimes.

### Three modelling choices a reviewer will ask about

1. **The policy acts on the windowed system-level signal, not a per-task draw.**
   Section II writes `v_k = κ u_k`, where `u_k` is a system-level quantity. The
   simulator therefore has the policy respond to the monitoring signal of the
   *previous closed window*. That one-window delay is what makes the loop a
   discrete-time map capable of period doubling; with a per-task instantaneous
   response the loop self-regulates continuously and no flip exists.

2. **The monitoring signal is measured error, not pre-verification confidence.**
   Section II writes `u_k` as system-level *error*. Using the agents'
   pre-verification confidence instead adds a second window of lag and turns the
   map into a two-step one, which produces **period-three** cycles rather than
   the predicted flip. This was observed before the change and is worth stating
   in the paper: the predicted failure mode is sensitive to where monitoring
   sits relative to actuation.

3. **The policy set-point tracks the operating point.** The local model is a
   linearization *around an operating point*; the simulator tracks it with an
   EWMA whose time constant (50 windows) is far longer than the window, so it
   cannot mask window-scale dynamics but does keep nominal effort at `v̄`.
   Without it, raising `κ` moves the operating point up the saturating efficacy
   curve instead of raising the loop gain.

### The indexing convention (`lead_inputs`)

In the paper the effort indexed `k` responds to the uncertainty indexed `k` and
acts on the transition into `k+1`. In the testbed the policy observes a closed
window and applies the effort during the *next* one, so recorded `v` and `q`
sit one index ahead. `identify(..., lead_inputs=True)` (the default) restores
the paper's convention:

```
u_{k+1} ~ [u_k, q_{k+1}, v_{k+1}]
q_{k+1} ~ [q_k, u_k,     v_{k+1}]
```

With the raw indices, `c` estimates as ≈ 0 with unstable sign, because the
contemporaneous effect of effort on error has already been absorbed into `u_k`.
**Which convention a deployment needs depends on where its monitoring sits
relative to its actuation, and it must be stated rather than assumed.**

---

## S6 — real LLM agents, a real shared verifier

S1–S5 run on the simulator. `s6_real_agents.py` asks S1's question of real
model calls on one laptop: does contention for a shared verification service
damage objective correctness, and does that depend on the fallback?

```bash
ollama pull qwen2.5:3b
python experiments/s6_real_agents.py        # 5 seeds, ~2 h, writes results/s6_real_agents.json
```

It is not part of `run_all.sh`: it needs a local model server and its
wall-clock is real time, not simulated time.

**What it is, precisely.** Concurrent task agents, each holding an answer
produced by a real LLM, competing for a real LLM verifier with one slot,
against synthetic competing traffic. It is a correction-service experiment.

| piece | what it is here |
|-------|-----------------|
| model | `qwen2.5:3b` via Ollama, temperature 0, one Apple M4 |
| task, scoring | `a × b`, `a` three-digit, `b` two-digit; exact integer match against `a*b` computed in Python; a reply with no number is an error (none occurred) |
| answering | one direct call per task, done **once per seed on an idle server before any arm**; the saved answers are replayed in every arm, so answering never competes with verification |
| uncertainty `û` | `1 − exp(Σ logprob)` of the answer tokens |
| agents | one per arriving task, Poisson arrivals at 0.62/s; 100 per arm |
| verifier | a second call to the same model that recomputes the product from two partial products without seeing the agent's answer; its answer replaces the agent's. No tool, no ground truth. 0.87 s per call |
| shared service | verifier pool, `C = 1` slot, FIFO (Ollama serves one request at a time) |
| background traffic | synthetic tenants, Poisson at 0.4/s, exponential hold with mean 1.0 s; they sleep in the slot and make no model calls. Low load = none; high load = on |
| policies | `always`; `proportional` `v = κû`; `aware` `v = κû/(1+γq)`; `v` read as verification probability; `κ = 4`, `γ = 1` |
| fallback | `wait`: no deadline. `skip`: a request not started within `D = 1.5 s` is **cancelled and leaves the queue** (it never occupies capacity) and the unverified answer is returned. A verification that has started is never cut off |
| controls | tasks, arrivals and verification draws matched across arms within a seed; arm order randomised per seed; pool drained between arms |

In the results, "completed" counts completed *verifications*; every task completes in
every arm. Waits of abandoned requests enter at the time they gave up.
Latency runs from an agent's arrival to its final answer and excludes the
precomputed answering call.

**Scope.** One model, one machine, one task family, low concurrency, synthetic
competing traffic, and no shared context store — so this tests the *direct*
fallback term of `b`, not the propagated one, and makes no claim about
`κ̂_crit`.

The design and decision rule are in `frozen/prereg.md` (addendum 2). Results:
`results/s6_real_agents.json`; an earlier single-seed pilot on `qwen3:8b` is
`results/s6_pilot_qwen3_8b.json`.

## S7 — closed loop on real agents

S6 has no feedback. `s7_real_loop.py` closes the loop on the same real
components and runs identify → freeze → sweep on it, reusing `ecfi.identify`
and `ecfi.detect` unchanged. Design, gate and crossing criterion are in
`frozen/prereg.md` (addendum 3), written before the runs.

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=2 ollama serve &
python experiments/s7_real_loop.py identify --ollama http://127.0.0.1:11435/api/chat   # ~2 h
python experiments/s7_real_loop.py sweep    --ollama http://127.0.0.1:11435/api/chat   # ~8 h
```

| piece | what it is here |
|-------|-----------------|
| window | `W = 40` completed tasks, ≈ 16 s; 15 windows discarded |
| uncertainty / error `u_k` | fraction of the window's final results wrong against ground truth (an audit signal) |
| effect on later demand | next window, each agent verifies with probability `v = sat_[0,1](0.4 + κ(u_k − u_ref))`, `u_ref` an EWMA of `u` (20 windows) |
| epistemic coupling | half the tasks add an upstream result read from a shared store of the last 40; wrong if either part is |
| congestion `q_k` | waiting requests per verifier slot, averaged over the window's arrivals |
| adjustable gain | `κ`, fixed per run |
| service capacity | `C = 2` live verifier slots (≈ 1.9 verifications/s), fixed; arrivals 2.5 tasks/s; deadline 1.5 s, then the unverified answer stands |
| real / not real | verification calls, queueing, timeouts, store and policy are live; direct answers are replayed from a bank answered once by the same model; background tenants are synthetic |

Results: `results/s7_identify.json`, `results/s7_sweep.json`. Frozen prediction:
`frozen/real_agents/`.

---

## Defaults

| symbol | meaning | default |
|--------|---------|---------|
| `n_agents` | agents `N` | 50 |
| `n_verifiers` | verifier servers `C` | 8 |
| `load_factor` | `λ v̄ / (C μ)` at nominal effort | 0.55 |
| `p0` | nominal error probability | 0.12 |
| `p_dep`, `n_dep`, `delta_dep` | context-store coupling | 0.80, 4, 0.45 |
| `beta`, `v_half` | verification efficacy | 2.0 / 14.0, 1.30 |
| `deadline` | `D`, on waiting for a verifier | 2.0 |
| `m_fallback` | fallback error multiplier | 2.0 |
| `W` | window size, tasks | 200 (400 for identification) |
| seeds per cell | | 20 |

`W` matters more than it looks: `u_k` carries binomial noise `√(u(1−u)/W)`, and
identification is noise-limited, which is why it uses wider windows.

## Reporting obligations

Section VI names three outcomes that count as evidence *against* the framework
and must be published if observed:

1. no correctness gap between `WAIT` and `RESOURCE_COUPLED` under load;
2. a transition far from κ̂_crit despite an adequate linear fit (R² is reported
   per equation);
3. absence of the H4 precursors on approach.

Poor identification fit is a scope limit of the local model, not support for it.
`s2_identify.py` prints a warning rather than failing silently when the fit is
weak or the excitation is poor.
