#!/bin/bash
# Pause the rigidity sweep after its current run, evaluate the trade-off runs
# W0, W0.0001, W0.0003, W0.001, W0.003 and the GT-mask runs G1, G2 on test, then resume the sweep (finished runs are skipped).
cd /home/iml/fryderyk.koegl/code/LapIRN-koegl/journal_results
DONE=rigidity_sweep/staring/val/staring_w1.98_lesions.csv
while [ ! -f "$DONE" ]; do sleep 30; done
sleep 30
tmux kill-session -t rigidity_sweep
sleep 30
cd ../Code
../.venv/bin/python -c 'import evaluate as e; e.SPLIT="test"; e.RUN_IDS=["W0","W0.0001","W0.0003","W0.001","W0.003","G1","G2"]; e.main()' 2>&1 | tee ../journal_results/test_tradeoff_low.log
cd ../journal_results
tmux new-session -d -s rigidity_sweep "cd /home/iml/fryderyk.koegl/code/LapIRN-koegl/journal_results && ../.venv/bin/python sweep_rigidity.py 2>&1 | grep --line-buffered -v -i warning | grep --line-buffered -v 'IO optimization' | tee -a sweep_rigidity.log"
