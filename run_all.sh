#!/usr/bin/env bash
# Full identify -> predict -> sweep sequence, in the order the method requires.
# Nothing here may be reordered: S3 reads a prediction that S2 froze.
set -euo pipefail

# same rule the Python side uses: CPUs this process is *allowed* to use,
# not the CPUs the machine happens to have
JOBS="${JOBS:-$(python3 -c 'from ecfi.runner import default_jobs; print(default_jobs())')}"
SEEDS="${SEEDS:-20}"
echo "running with JOBS=$JOBS SEEDS=$SEEDS"

python3 -m pytest -q

echo; echo "### S1  causal go/no-go for b > 0"
python3 experiments/s1_gonogo.py --jobs "$JOBS" --seeds "$SEEDS"

for REGIME in steep shallow; do
  echo; echo "### S2  identification and freeze  [$REGIME]"
  python3 experiments/s2_identify.py --regime "$REGIME" --jobs "$JOBS" --seeds "$SEEDS"

  echo; echo "### S3  out-of-sample sweep  [$REGIME]"
  python3 experiments/s3_sweep.py --regime "$REGIME" --jobs "$JOBS" --seeds "$SEEDS"
done

echo; echo "### endpoints"
python3 experiments/analyze_s3.py

echo; echo "### S4  strategic pressure"
python3 experiments/s4_strategic.py --jobs "$JOBS" --seeds "$SEEDS"

echo; echo "### S5  cross-layer mitigation"
python3 experiments/s5_mitigation.py --jobs "$JOBS" --seeds "$SEEDS"

echo; echo "### figures"
python3 experiments/make_figures.py

echo; echo "done. results/ and figs/ are populated."
