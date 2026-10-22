#!/bin/bash
# Rigidity sweep on janus (from the front) and on the cluster (from the back):
# every 5 min, copy finished cluster runs into the local sweep folder (the local
# sweep skips runs whose CSV exists). Once all 15 runs exist, stop the local
# sweep and compute curve.csv / chosen.json (sweep_rigidity.py, which then only
# reads the existing CSVs).
cd /home/iml/fryderyk.koegl/code/LapIRN-koegl/journal_results
REMOTE=fryderyk.koegl@hpc-submit02.scidom.de:/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval/rigidity_sweep/
while true; do
    rsync -a "$REMOTE" rigidity_sweep/ 2>/dev/null
    n=$(ls rigidity_sweep/*/val/*_w*.csv 2>/dev/null | grep -vc lesions)
    echo "$(date +%H:%M) $n/15 runs"
    [[ $n -ge 15 ]] && break
    sleep 300
done
tmux kill-session -t rigidity_sweep 2>/dev/null
sleep 30
../.venv/bin/python sweep_rigidity.py 2>&1 | grep -v -i warning | tee sweep_rigidity_final.log
