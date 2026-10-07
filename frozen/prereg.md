# Pre-registration — ECFI benchmark

Fill this in and commit it **before** running `s2_identify.py`. Once
`frozen/<regime>/params.json` exists, this document and those files are the
record that the prediction preceded the test. Do not edit either afterwards;
if the design must change, start a new dated section and say why.

---

## 1. Scope

Testbed: SimPy discrete-event simulation in `ecfi/sim.py`. Regimes swept:
`steep` and `shallow` (`experiments/regimes.py`). All parameters not listed as
swept take the defaults in `SimConfig`.

Replications: **20 seeds per cell**, seeds `1…20`, fixed before any run.
Windows per run: 300 (S1/S3/S4/S5), 420 (S2). Warm-up discarded: 60 windows.

## 2. S1 — causal go/no-go for `b > 0`

**Design.** Randomised block order over background-load levels
`{0, 1, 2, 3}`, 40 windows per block, order permuted per seed. Three arms:
`WAIT`, `RESOURCE_COUPLED`, and `RESOURCE_COUPLED` with the shared context
store disabled. Tasks carry time-insensitive ground truth, so waiting alone
cannot stale the evidence.

**Primary quantity.** `Δ_error(arm) = error(load 3) − error(load 0)`.

**Decision rule — pre-specified.** The ECFI pathway is declared supported iff

```
Δ_error(RESOURCE_COUPLED) − Δ_error(WAIT)  >  0.005
```

and the difference is significant at α = 0.05 by a two-sided Welch t-test over
seed-level estimates. The 0.005 threshold is one-half of the smallest error
change considered operationally meaningful (0.01, i.e. one extra error per 100
tasks).

The seed is the unit of replication: `Δ_error` is computed *per seed* and the
test runs over those values, never over pooled windows, which would understate
the variance and inflate significance. `s1_gonogo.py` evaluates this rule and
prints PASS/FAIL for each half of it; it exits non-zero when the pathway is not
supported, so the stop condition below cannot be skipped by accident.

**Secondary.** `Δ_error` with the store disabled isolates the *direct* fallback
term; the difference between the two coupled arms is the *propagated* term.
Both are reported.

**Stop condition.** If the rule fails, S3–S5 are not run as boundary tests. The
study is reported as a negative result plus a map of the conditions under which
the coupling does and does not appear (see branch A in the plan).

## 3. S2 — identification and freeze

**Excitation, all three simultaneously, at mutually non-commensurate periods:**

| target | intervention | setting |
|--------|--------------|---------|
| `a`, `e` | ambiguity bursts | period 23 windows, duty 0.30, level 0.80 |
| `b`, `d` | background-load pulses and their decay | period 17 windows, duty 0.40, level 3.0 |
| `c`, `f` | randomised **nominal effort** `v̄` | block 5 windows, uniform ±0.60 |

Dithering `v̄` rather than `κ` is deliberate: near the operating point
`u − u_ref` is small, so a change in `κ` barely moves `v` and leaves the design
near-collinear.

**Estimator.** OLS per equation on windowed deviations, `lead_inputs=True`
(convention stated in the README). Moving-block bootstrap, 600 resamples,
block length set from the residual correlation time, 95% percentile intervals.

**Validity gate — pre-specified.** The identification is usable only if
`R²(u) > 0.10`, `R²(q) > 0.10`, and `|corr(u,v)| < 0.95`. Otherwise the result
is reported as a scope limit of the local model, **not** as support for it, and
no boundary claim is made for that regime.

**Freeze.** `κ̂_crit` and its CI are written to `frozen/<regime>/kcrit.json`
with a UTC timestamp and git commit, then committed and tagged.

## 4. S3 — out-of-sample boundary test

**Grid.** `κ ∈ linspace(0, 1.5 κ̂_crit, 13)`; load factors `{0.45, 0.55, 0.65}`.
Fixed before the freeze is read.

**Primary endpoint — threshold-free, two independent estimators:**

1. **Margin knee.** The smallest `κ` at which the online margin
   `M = 1 − ρ(Â)` has come within 5% of its total descent,
   `M(κ*) ≤ M_min + 0.05 (M(0) − M_min)`, linearly interpolated.
2. **Variance extrapolation.** Linear fit of `1/var_ratio(u)` over the rising
   portion (first 60% of its descent), solved for its zero.

Both are reported. A stochastic nonlinear system does not produce a clean limit
cycle: saturation caps the critical mode short of −1, so `M` flattens at a
positive floor. These endpoints measure where the eigenvalue has *arrived*
rather than requiring it to reach exactly −1.

**Reported statistic.** `κ* / κ̂_crit`, per load factor, mean and sd across
load factors.

**Explicitly not used as the primary endpoint:** a spectral threshold
calibrated on the store-disabled control. Disabling the store removes the
epistemic propagation path but leaves the corrective feedback loop intact, so
that arm alternates too and cannot serve as a null for the flip. It is retained
as a secondary readout only, and this limitation is reported.

**H4 precursors logged on approach:** windowed variance, lag-one
autocorrelation of `u_k`, recovery time after a standardised load pulse, and
`M(t)`.

## 5. S4 — strategic pressure

Strategic tasks *replace* benign ones at fixed arrival rate,
`ε ∈ {0, 0.05, 0.10, 0.20}`, at `κ ∈ {0.25, 0.5, 0.75, 1.0} × κ̂_crit`.
Baseline and strategic runs are **seed-matched**.

`amb_true_boost = 0`: strategic inputs raise apparent uncertainty without
raising ground-truth difficulty, isolating exploitation of the policy from
simply sending harder work.

`𝒜_E = (ΔR/R_base) / (ΔU_inj/U_base)` reported for three resource measures
(effort served, queue occupancy, p95 wait). `ΔU_inj` is the realised strategic
fraction times the apparent-uncertainty boost per task, normalised by the
baseline epistemic level.

**Pre-specified question.** Does `𝒜_E` rise as `κ → κ̂_crit`? Reported as the
curve, not as a single threshold test.

## 6. S5 — cross-layer mitigation

`γ ∈ {0, 5, 10, 25, 50, 100, 200}` at `κ = 1.25 κ̂_crit`, load factor 0.65.
Four axes reported together: objective error, p95 wait, effort served, and
stability (`M` plus alternation amplitude).

**Pre-specified success criterion.** A `γ` counts as an improvement only if it
raises `M` above the `γ = 0` value **while** costing no more than 2% in
objective error. If no `γ` meets it, that is reported — a controller that buys
stability by verifying less has not solved the problem.

## 7. Outcomes that count against the framework

Published if observed, per Section VI:

1. no correctness gap between `WAIT` and `RESOURCE_COUPLED` under load;
2. a transition far from `κ̂_crit` despite an adequate linear fit;
3. absence of the H4 precursors on approach.

Per-equation `R²` is reported in all cases.

---

## Sign-off

| field | value |
|-------|-------|
| Registered by | |
| Date (UTC) | |
| Git commit at registration | *(fill in: `git rev-parse HEAD` before running S2)* |
| Git commit at freeze | *(recorded automatically in `kcrit.json`)* |
| Deviations from this document | *(none, or list with reasons and dates)* |

### How to register

```bash
git add frozen/prereg.md
git commit -m "Pre-register ECFI benchmark S1-S5"
git tag -a prereg -m "design fixed before identification"
git rev-parse HEAD          # paste into the table above, amend the commit
```

Then, after `s2_identify.py` writes the freeze:

```bash
git add frozen/
git commit -m "Freeze kappa_hat_crit for steep and shallow"
git tag -a freeze -m "prediction fixed before the S3 sweep"
```

Two tags, in that order, in the repository history are what make the ordering
auditable by someone who was not there.

---

## Addendum, 2026-10-05 — S6, real-agent experiment (added in revision)

Added in response to review ("test real agents and shared services, including
alternative verification policies and fallback behavior"). Sections 1–7 are
unchanged. Written before the full run; what preceded it is listed under
*Disclosure*.

**Testbed.** `experiments/s6_real_agents.py`. Agents are `qwen3:8b` served by
Ollama on one Apple M4 (16 GB), temperature 0. Tasks: `a × b`, `a` four-digit,
`b` two-digit, scored by exact match. Verification is a step-by-step
recomputation by the same model, routed through a verifier pool with `C = 1`
slot. Background load is synthetic tenants holding the slot (mean 5 s) at
utilisation `{0, 0.4}`.

**Design.** 50 tasks per arm, Poisson arrivals with mean gap 10 s, seed 1.
Tasks, arrival gaps and verification draws are identical across arms. Arms:
`always/wait`, `always/skip`, `proportional/skip` (`κ = 4`), `aware/skip`
(`κ = 4`, `γ = 1`), each at both load levels. Deadline `D = 8 s` on waiting for
a slot.

**Reported for every arm.** Objective error, unverified (primary) error,
verification calls requested and completed, timeouts, queue wait (mean, p95),
queue length seen at arrival, service time, task latency (mean, p95).

**Primary quantity and decision rule.** With always-verify, per task,

```
DiD = [wrong(skip, high) − wrong(skip, low)] − [wrong(wait, high) − wrong(wait, low)]
```

The congestion → correctness pathway is declared present in real agents iff
`mean(DiD) > 0.05` and `mean(DiD) > 1.96 · se(DiD)`. Otherwise it is reported
as not detected at this sample size. A null here counts under outcome 1 of
section 7.

**Secondary, descriptive only.** Whether `proportional` and `aware` reach a
lower error than `always/skip` under high load, and at how many verification
calls.

**Scope.** One model, one machine, one seed, one task family, no shared
context store: this tests the *direct* fallback term of `b`, not the
propagated term, and makes no claim about `κ̂_crit` or the flip.

**Disclosure.** Before this addendum: ~30-task pilots per task difficulty to
pick a task where direct and verified accuracy differ and to measure service
time (direct ≈ 0.67 correct, verified ≈ 1.0, verify ≈ 5.5 s), and one 8-task
smoke run of all arms to check the code. `D`, the arrival rate and the
background level were set from those service times; `κ` and `γ` were not tuned
on outcomes.

---

## Addendum 2, 2026-10-05 — S6 redesigned after the first run

The run registered above was executed (`results/s6_pilot_qwen3_8b.json`:
DiD = +0.120 ± 0.046 within-run se, 50 tasks, one seed). It is retained and
reported as a **pilot**, not as the S6 result, because review of it identified
three design faults: arm order was fixed (low load always before high),
verification calls delayed the answering calls on the same server, and a single
run gives no between-run uncertainty. The design below replaces it and was
written before any run of it beyond the smoke test disclosed at the end.

**Changes.**

1. *Answering isolated from verification.* Each seed's tasks are answered once,
   serially, on an idle server, before any arm. The saved answers and
   uncertainties are replayed in every arm. S6 is therefore a
   correction-service experiment: concurrent agents holding real model answers
   compete for a real LLM verifier.
2. *Arm order randomised per seed*; the verifier pool is drained (queued work
   cancelled, slot idle) and the server rested 2 s between arms.
3. *Five independent runs*, seeds 1–5. Each seed draws new tasks, arrival
   times, verification draws, arm order and background traffic.
4. *Faster model and verifier*, to make five runs feasible: `qwen2.5:3b`;
   tasks `a × b` with `a` three-digit, `b` two-digit; the verifier writes the
   two partial products and their sum (three lines). 100 tasks per arm, mean
   arrival gap 1.6 s, deadline `D = 1.5 s`, background tenants with mean hold
   1.0 s at utilisation 0.4 (0.4 tenants/s). `C = 1`, `κ = 4`, `γ = 1`
   unchanged. Arms unchanged.

**Primary quantity and decision rule.** The seed is the unit of replication.
Per seed, `DiD` as defined above (always-verify, skip vs wait, high vs low
load, mean over that seed's paired tasks). The pathway is declared present iff
the mean of the five seed-level values exceeds 0.05 **and** differs from zero
at α = 0.05 by a two-sided one-sample t-test (4 df). Every seed's value is
reported. Five seeds are a practical minimum, not a power calculation.

**Reported per arm, mean and sd across seeds.** Everything listed in the first
addendum, plus observed agent concurrency (time-average and maximum), realised
background occupancy of the slot, agent occupancy of the slot, and the counts
of answers the verifier fixed, broke, or failed to return.

**Secondary, descriptive only.** As before.

**Scope.** As before, and narrower in one respect: answers are precomputed, so
this does not measure contention on the answering path. No claim about
feedback instability is made from S6.

**Disclosure.** Before this addendum, on a pilot seed (123) that is not a study
seed: ~30 tasks at each of three difficulties to choose one where direct and
verified accuracy differ (three-digit × two-digit: direct ≈ 0.30 correct,
verified ≈ 0.70–0.77), and a comparison of two verifier prompts for service
time (≈ 3.3 s vs ≈ 0.95 s at equal accuracy; the faster was kept). Then one
smoke run of the revised script, seeds 1–2 with 15 tasks per arm, to check the
code; its first 15 tasks per seed coincide with the first 15 of the study's
100. `D`, the arrival gap and the tenant hold time are the first run's values
rescaled to the new service time; they were not tuned on outcomes.

---

## Addendum 3, 2026-10-05 — S7, closed-loop real-agent boundary test

Added in response to review: implement the feedback loop on real agents,
identify its parameters separately, freeze the predicted boundary, then test it
in independent gain-sweep runs. Written before any identification or sweep run;
what preceded it is under *Disclosure*. Script: `experiments/s7_real_loop.py`.
All constants below are fixed in that file.

**Testbed.** Same real components as S6 (addendum 2): `qwen2.5:3b` at
temperature 0, tasks `a × b` scored by exact match, a second LLM call as
verifier, a shared verifier pool with a deadline, synthetic background tenants.
Differences: `C = 2` slots on a dedicated Ollama server with parallel decoding
(measured capacity ≈ 1.9 verifications/s); arrivals Poisson at 2.5 tasks/s;
deadline 1.5 s on the wait, timed-out requests cancelled, unverified answer
returned. Direct answers come from a bank of 2000 three-digit and 600
four-digit tasks answered once on an idle server; each run samples the bank
with replacement. Verification calls, queueing, timeouts, the store and the
policy are live.

**The loop.**

* Window: `W = 40` completed tasks (≈ 16 s). First 15 windows discarded.
* `u_k`: fraction of window `k`'s final results that are wrong against ground
  truth (an audit signal; the same observable the simulator's policy uses).
* Demand: in the following window each arriving agent verifies with
  probability `v = sat_[0,1](v̄ + κ (u_k − u_ref))`, `v̄ = 0.4`, `u_ref` an EWMA
  of `u` with time constant 20 windows. `κ` is the adjustable gain.
* Epistemic coupling: with probability 0.5 a task's result is its own product
  plus an upstream product read from a shared store of the last 40 results;
  it is wrong if either addend is.
* `q_k`: mean over the window's arrivals of waiting requests per slot.
* Service capacity is fixed at `C = 2`; congestion is varied through
  background occupancy, not through `C`.

**Identification.** Three runs, seeds 101–103, 135 windows each, `κ = 1`, all
three excitations at once, indexed by window:

| target | intervention | setting |
|--------|--------------|---------|
| `a`, `e` | hard-task bursts (ambiguity) | 60% four-digit tasks for 3 of every 7 windows |
| `b`, `d` | background-load pulses | tenants at 0.5 of capacity for 5 of every 11 windows |
| `c`, `f` | randomised nominal effort `v̄` | uniform 0.4 ± 0.25, redrawn every 3 windows |

Estimator as in section 3: OLS per equation on windowed deviations,
`lead_inputs=True`, moving-block bootstrap with 600 resamples for the 95%
interval of `κ̂_crit`.

**Validity gate.** `R²(u) > 0.10`, `R²(q) > 0.10`, `|corr(u,v)| < 0.95`;
leave-one-run-out `R² > 0` for both equations in every fold; the fitted model
Schur-stable at `κ = 0`; `b, c, e, f > 0`; `κ̂_crit` finite and positive. If any
part fails, that is reported as a scope limit of the local model on this
system and no claim of support is made, whatever the sweep shows.

**Freeze.** `frozen/real_agents/kcrit.json`: `κ̂_crit`, its interval, the gate
outcome, and the sweep grid `κ ∈ {0, 0.5, 0.8, 1.0, 1.25, 1.6} × κ̂_crit`. If
`κ̂_crit` is not finite and positive the grid is `{0, 1, 2, 3, 4.5, 6}` and the
sweep is descriptive only. The sweep phase is started by the same unattended
command immediately after the freeze is written and reads the grid from it; no
sweep result exists or is seen before the freeze.

**Sweep.** Seeds 1–3 at each of the six gains: 18 runs of 95 windows, no
excitation, run order randomised over gain and seed.

**Early-warning metrics, per gain, mean and se over seeds.** Variance of `u`
relative to `κ = 0`; lag-one autocorrelation of `u`; Nyquist fraction of `v`
(spectral); fitted stability margin `M = 1 − ρ(Â)`, set against the margin the
frozen parameters predict at that gain.

**Crossing criterion.** A gain counts as past the boundary iff the lag-one
autocorrelation of `u` is negative with its 95% t-interval over seeds
excluding zero **and** the variance ratio is at least 2. If no swept gain
meets it, *absence of crossing* is reported.

**Endpoint and agreement.** Primary endpoint: margin knee (section 4).
Secondary: variance extrapolation, which is fragile on a six-point grid. The
prediction is said to agree iff a crossing is observed and
`κ*/κ̂_crit ∈ [0.5, 1.5]`; otherwise disagreement is reported with the ratio.

**Known limits, stated in advance.** `v` is a probability, so the actuator
saturates at 0 and 1; with `W = 40` the window noise in `u` is about 0.08, so
near the predicted boundary the loop will often be saturated and the
transition may be smeared or capped. Three seeds per gain and 80 windows per
run give coarse estimates. One model, one machine, one task family.

**Disclosure.** Before this addendum: a throughput test of the parallel server
(1, 2, 4 concurrent verifications); an offline check of the analysis code on
synthetic autoregressive series; and one smoke run of the identification phase
on seeds 901–902 with 27 windows, written to a scratch directory, to check the
code. No gain other than `κ = 1` has been run. `V̄`, the arrival rate and the
window were set from the measured service time and capacity, not from loop
outcomes.

*Smoke outcome, recorded before the study runs.* The smoke run completed: 12
post-warm-up windows per seed at ≈ 16 s per window, error level `u ≈ 0.68–0.77`,
timeouts 12–19% of tasks. Its 24-row fit is not meaningful and was not used.
No constant was changed after it. The high error level means `u` responds
weakly to `v` relative to window noise, which sharpens the saturation caveat
above.
