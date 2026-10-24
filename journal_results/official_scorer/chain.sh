#!/bin/bash
# Run the remaining stages unattended on janus: wait until a stage is complete
# (all result CSVs on lustre and janus' worker finished), make the choice the
# next stage needs (select_stage.py), then submit the next stage on the cluster
# and start janus' worker on it.
#
#     tmux new-session -d -s official_chain "bash chain.sh 2 2>&1 | tee chain.log"
#
# The argument is the stage that is currently running (2 or 3).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=../../.venv/bin/python
HOST=fryderyk.koegl@hpc-submit02.scidom.de

complete() {  # stage -> 0 if every task's result CSV is on lustre
    $PY - "$1" <<'EOF'
import subprocess, sys
import worker, tasks  # worker first: it puts Code/ on the path
stage = int(sys.argv[1])
reference = tasks.rigidity_reference(worker.REPO)
specs = tasks.STAGES[stage] + tasks.STAGE_WORKSTATION_FIRST.get(stage, [])
paths = " ".join(str(tasks.result_csv(s, worker.LUSTRE_OUT, reference)) for s in specs)
missing = subprocess.run(
    ["ssh", "-o", "BatchMode=yes", worker.HOST, f"for f in {paths}; do test -f $f || echo $f; done"],
    capture_output=True, text=True,
).stdout.split()
print(f"stage {stage}: {len(specs) - len(missing)}/{len(specs)} done", flush=True)
sys.exit(0 if not missing else 1)
EOF
}

wait_for() {  # stage
    until complete "$1" && ! tmux has-session -t "official_stage$1" 2>/dev/null; do
        sleep 600
    done
}

start() {  # stage
    bash submit_stage.sh "$1"
    tmux new-session -d -s "official_stage$1" \
        "cd $PWD && $PY worker.py janus $1 2>&1 | tee stage$1_janus.log"
    echo "$(date): stage $1 started"
}

stage=$1
if [[ $stage -le 2 ]]; then
    wait_for 2
    $PY select_stage.py pet || { echo "selection after stage 2 failed"; exit 1; }
    start 3
    stage=3
fi
if [[ $stage -le 3 ]]; then
    wait_for 3
    $PY select_stage.py rigidity || { echo "selection after stage 3 failed"; exit 1; }
    start 4
fi
echo "$(date): chain done (stage 4 running)"
