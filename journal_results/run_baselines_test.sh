#!/usr/bin/env bash
# Baselines on test (todo step 8): register the val-chosen NiftyReg / ConvexAdam
# configs on the test pairs, then evaluate B1, B1-floor, B2, B3, B4 on test
# (results: evaluation/test/<run>.csv). Finished pairs / runs are skipped.
#   tmux new-session -d -s baselines_test "bash run_baselines_test.sh"
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
$PY tune_convexadam.py --split test 2>&1 | grep --line-buffered -v -i warning | tee test_convexadam.log
$PY tune_niftyreg.py --split test 2>&1 | grep --line-buffered -v -i warning | tee test_niftyreg.log
cd ../Code
$PY -c "
import torch, evaluate
device = torch.device('cuda:0')
for run_id in ('B1', 'B1-floor', 'B2', 'B3', 'B4'):
    evaluate.evaluate_run(run_id, 'test', device)
" 2>&1 | grep --line-buffered -v -i warning | tee ../journal_results/test_baselines_eval.log
