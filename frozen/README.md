# frozen/

This directory is the pre-registration record.

* `prereg.md` — the design, decision rules and endpoints, committed **before**
  identification is run.
* `<regime>/params.json` — the identified local parameters, their bootstrap
  intervals, per-equation R², and the excitation diagnostics.
* `<regime>/kcrit.json` — the derived `κ̂_crit`, its CI, and the sweep range.

Both JSON files carry a UTC timestamp and the git commit they were produced at.

**Commit and tag this directory before running `s3_sweep.py`.** The claim the
paper makes is not that a fitted model reconstructs the transition — it is that
a prediction fixed in advance located it. That claim is only auditable if the
ordering is in the repository history.

`s2_identify.py` refuses to overwrite an existing freeze without `--refreeze`.
Passing that flag restarts the whole identify–predict–sweep sequence; it does
not "update" a prediction.

**This directory ships empty of freezes, on purpose.** A freeze is a record of
*your* prediction, made on *your* machine, at the seed count the protocol asks
for. Shipping one with the code would let someone else's number stand in for
yours the moment `s3_sweep.py` reads it, and the guard in `s2_identify.py` would
then read as an obstacle rather than as the protection it is.

Reference numbers from the run the testbed was developed on are kept in
`examples/demo-run-8seeds/`, where no script can reach them.
