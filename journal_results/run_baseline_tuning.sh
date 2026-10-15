#!/usr/bin/env bash
# Tune both baselines on val (ConvexAdam first: ~35 min; NiftyReg: ~7-8 h).
# Registered fields and evaluations are kept, so a rerun resumes where it stopped.
#   tmux new-session -d -s tune_baselines "bash run_baseline_tuning.sh"
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY tune_convexadam.py 2>&1 | grep --line-buffered -v -i warning | tee tune_convexadam.log
$PY tune_niftyreg.py 2>&1 | grep --line-buffered -v -i warning | tee tune_niftyreg.log
