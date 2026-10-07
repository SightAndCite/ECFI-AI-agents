# Step 1 — closed-loop Methods details (Section III-C, AUTHOR CHECK C5)

Answers to the six questions, taken from the script that produced the results
(`experiments/s7_real_loop.py`) and the saved data.

Three points need a wording change in the manuscript:

- the controller is an oracle (Q3);
- the timeout percentages are per task, not per verification request (Q4);
- the implemented gain is 5% below the nominal value (Q2).

## 1. The two measured states

- **Epistemic state `u_k`:** the fraction of the window's 40 final results that
  are wrong, scored by exact match against ground truth computed in Python. It
  is objective error, not model confidence or any estimate.
- **Congestion state `q_k`:** the number of verification requests waiting for a
  slot, divided by the number of slots (2), read at the moment each agent
  arrives and averaged over the window's 40 tasks.
- **Also recorded per window:** `v_k`, the mean verification probability
  applied to those tasks. It is the regressor for the effort coefficients.
- **Window definition:** a window closes when 40 tasks have completed, so tasks
  are binned by completion, not arrival. A few tasks arrive in one window and
  complete in the next.

## 2. The update equation

For each arriving agent, during the window after window `k` closes:

```
v = clip( 0.4 + κ · (u_k − ū_k), 0, 1 )
ū_k = ū_{k−1} + (u_k − ū_{k−1}) / 20        (ū_1 = u_1)
```

- **Baseline:** 0.4.
- **Clipping:** to [0, 1], since `v` is a probability. Each agent then verifies
  if an independent uniform draw is below `v`.
- **Smoothing:** none on `u_k` itself. The set-point `ū` is an exponential
  average with a 20-window time constant.
- **Delay:** one window. The policy reads only the most recently closed
  window's error; nothing older enters except through `ū`.
- **Inputs:** only `u_k`. Queue length and the task's own confidence do not
  enter (the congestion term is switched off).
- **Start-up:** `v = 0.4` until the second window closes.
- **Effective gain:** the set-point is updated with `u_k` before the deviation
  is taken, so the deviation is `0.95 · (u_k − ū_{k−1})`. The gain applied is
  therefore 5% below the nominal `κ` reported. The same code ran in
  identification and sweep.

## 3. Oracle controller

Yes, and it should be stated as such. The policy is driven by ground-truth
error, which is only available because the tasks are objectively scored. This
was intentional, to match the simulator, whose policy also acts on measured
error. No observable proxy was used: the logprob-based confidence from the
open-loop experiment plays no part in this one. A deployed system would need an
audit sample or a proxy such as verifier disagreement, and that was not tested.

## 4. Timeouts at zero gain

- **Baseline verification:** at zero gain, `v = 0.4` for every task, so 40% of
  tasks still request verification.
- **Why some time out:** that is about 1.0 request per second against a
  measured capacity of about 1.9 per second (two slots), so utilisation is
  about 0.53. Poisson arrival bursts occasionally make a request wait past the
  1.5 s deadline.
- **Denominator:** the reported timeout rate is timed-out tasks divided by all
  tasks in the window, not by verification requests. This differs from the
  open-loop experiment, which reported timeouts per request.

| Gain  | Requests ÷ tasks | Timeouts ÷ tasks (as reported) | Timeouts ÷ requests |
|-------|------------------|--------------------------------|---------------------|
| 0     | 0.40             | 0.020                          | 0.049               |
| 4.12  | 0.40             | 0.067                          | 0.167               |
| 6.59  | 0.52             | 0.147                          | 0.283               |
| 8.24  | 0.46             | 0.139                          | 0.300               |
| 10.30 | 0.47             | 0.140                          | 0.300               |
| 13.18 | 0.47             | 0.159                          | 0.335               |

A timed-out request is cancelled before it starts, never occupies a slot, and
the agent returns its unverified answer.

## 5. Windows and seeds

- **Sweep runs:** 95 windows collected, the first 15 discarded, 80 analysed.
- **Identification runs:** 135 collected, 15 discarded, 120 analysed, giving
  357 regression rows over three runs.
- **Seeds:** identification uses seeds 101–103; the sweep uses seeds 1–3 at
  each of six gains (18 runs, order randomised over gain and seed).
- **Summary across seeds:** every metric is computed per run, then averaged
  over the three runs. The ± values are 95% t-intervals with 2 degrees of
  freedom.
- **Variance ratio:** a run's variance of `u` divided by the mean variance of
  the three zero-gain runs.
- **Fitted margin:** per run, the mean of a sliding 40-window estimate; then
  the mean over seeds.

## 6. Identification, held-out split and the interval

All three interventions run at once, at gain 1:

| Coefficients | Intervention               | Setting                                                         |
|--------------|----------------------------|-----------------------------------------------------------------|
| `a`, `e`     | hard-task bursts           | 60% four-digit tasks for 3 of every 7 windows                   |
| `b`, `d`     | background-load pulses     | synthetic tenants at 0.5 of capacity for 5 of every 11 windows  |
| `c`, `f`     | randomised baseline effort | uniform 0.4 ± 0.25, redrawn every 3 windows                     |

- **Estimator:** ordinary least squares per equation, on deviations from each
  run's own mean: `u_{k+1} ~ [u_k, q_{k+1}, v_{k+1}]` and
  `q_{k+1} ~ [q_k, u_k, v_{k+1}]`. Rows never cross a run boundary.
- **Held-out split:** leave one run out. Fit on two runs, compute R² on the
  third, for each of the three folds.
- **Interval for 8.24:** a moving-block bootstrap over the stacked regression
  rows, 600 resamples. Block length is twice the residual correlation time,
  with a minimum of 4 rows. The critical gain is recomputed from each
  resample's coefficients, and the 2.5th and 97.5th percentiles give 5.42 to
  17.48.
- **Caveat on the interval:** bootstrap blocks are drawn from the stacked rows,
  so a block can span two runs.

## What this means for the manuscript

It is a reproducible closed-loop test in the sense that the script, constants,
frozen prediction and per-window data are all in the repository:

- **Script and constants:** `experiments/s7_real_loop.py`
- **Frozen prediction:** `frozen/real_agents/`
- **Design:** `frozen/prereg.md`, addendum 3
- **Data:** `results/s7_identify.json` and `results/s7_sweep.json`

What it demonstrates is narrower than a boundary validation: an oracle-driven,
probability-capped controller on real verification calls, whose identification
failed its validity check and which showed no crossing.
