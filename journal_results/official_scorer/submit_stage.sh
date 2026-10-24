#!/bin/bash
set -euo pipefail

# Usage (on janus): submit_stage.sh STAGE
# Only some tasks of the stage: ARRAY=46,47,49 submit_stage.sh STAGE
# Copies Code/ and journal_results/official_scorer/ to a snapshot on the cluster
# and submits worker_array.sbatch with one array task per task of the stage
# (tasks.STAGES), without an own concurrency limit (Slurm enforces the user's).
[[ $# -eq 1 ]] || { echo "usage: $0 STAGE" >&2; exit 1; }
STAGE=$1
HOST="${HOST:-fryderyk.koegl@hpc-submit02.scidom.de}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
N=$(cd "$HERE" && "$REPO/.venv/bin/python" -c "
import sys; sys.path.insert(0, '$REPO/Code')
import tasks; print(len(tasks.STAGES[$STAGE]))" 2>/dev/null | tail -1)
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP="/home/iml/fryderyk.koegl/code_snapshots/official_stage${STAGE}_$STAMP"

ssh "$HOST" "mkdir -p $SNAP"
rsync -a --exclude='__pycache__' --exclude='*.pyc' "$REPO/Code/" "$HOST:$SNAP/Code/"
rsync -a --exclude='__pycache__' "$HERE/" "$HOST:$SNAP/official_scorer/"
git -C "$REPO" rev-parse HEAD | ssh "$HOST" "cat > $SNAP/.snapshot_git_head"
git -C "$REPO" diff HEAD | ssh "$HOST" "cat > $SNAP/.snapshot_git_diff"

JOBID=$(ssh "$HOST" "sbatch --parsable --array=${ARRAY:-0-$((N - 1))} \
    --export=ALL,CODE_ROOT=$SNAP,STAGE=$STAGE $SNAP/official_scorer/worker_array.sbatch")
echo "Submitted stage $STAGE (tasks ${ARRAY:-0-$((N - 1))} of $N) as array $JOBID, snapshot $SNAP"
