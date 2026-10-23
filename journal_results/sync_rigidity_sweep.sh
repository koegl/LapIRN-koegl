#!/bin/bash
# Rigidity sweep on janus (from the front) and on the cluster (from the back):
# every 5 min, copy finished cluster runs (sweep, val reference I2, test R1-R4)
# into the local folders (the local
# sweep skips runs whose CSV exists). Once all runs exist, stop the local
# sweep and compute curve.csv / chosen.json (sweep_rigidity.py, which then only
# reads the existing CSVs). Start it only after the cluster runs are submitted:
# sweep_rigidity.py would compute missing runs locally.
cd /home/iml/fryderyk.koegl/code/LapIRN-koegl/journal_results
N_RUNS=${N_RUNS:-19}  # 5 Staring + 7 Staring + erosion + 7 Kabsch
CLUSTER=fryderyk.koegl@hpc-submit02.scidom.de:/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval
while true; do
    rsync -a "$CLUSTER/rigidity_sweep/" rigidity_sweep/ 2>/dev/null
    # reference R1 on val (= I2) and the test runs R1-R4
    rsync -a "$CLUSTER/evaluation/val/" ../evaluation/val/ 2>/dev/null
    rsync -a "$CLUSTER/evaluation/test/" ../evaluation/test/ 2>/dev/null
    n=$(ls rigidity_sweep/*/val/*_w*.csv 2>/dev/null | grep -vc lesions)
    echo "$(date +%H:%M) $n/$N_RUNS runs, val reference: $([[ -f ../evaluation/val/I2.csv ]] && echo yes || echo no)"
    [[ $n -ge $N_RUNS && -f ../evaluation/val/I2.csv ]] && break
    sleep 300
done
tmux kill-session -t rigidity_sweep 2>/dev/null
sleep 30
../.venv/bin/python sweep_rigidity.py 2>&1 | grep -v -i warning | tee sweep_rigidity_final.log
