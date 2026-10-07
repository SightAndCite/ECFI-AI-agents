# Demonstration run — 8 seeds, 2 cores

Reference output from the machine the testbed was written on, kept so you can
sanity-check that your own run lands in the same neighbourhood.

**No script reads anything in this directory.** It is deliberately outside
`frozen/` and `results/`, because a freeze belonging to someone else's run must
never be able to stand in for yours: `s3_sweep.py` reads
`frozen/<regime>/kcrit.json`, and if that file were this one, the out-of-sample
claim would be tested against a prediction you did not make.

Seeds: 8 (the protocol asks for 20). Treat every number here as indicative, not
quotable.

| | steep | shallow |
|---|---|---|
| κ̂_crit | 4.281 | 29.747 |
| 95% CI | [4.07, 4.50] | [26.32, 33.60] |
| observed κ*/κ̂_crit | 0.744 ± 0.021 | 0.470 ± 0.069 |
| R²(u) / R²(q) | 0.44 / 0.50 | 0.52 / 0.63 |

S1 at 8 seeds: Δ_error Wait −0.0007, Resource-coupled +0.0112, store disabled
+0.0054.
