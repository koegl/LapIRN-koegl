#!/bin/bash
set -euo pipefail

# Usage (on janus): submit_eval_array.sh SPLIT RUN_ID [RUN_ID ...]
# Copies Code/ to a timestamped snapshot on the cluster and submits
# psmareg_eval_array.sbatch as an array with one evaluation run per task.
# Results: $PSMAREG_DATA_PATH/evaluation/<split>/ on the cluster.
[[ $# -ge 2 ]] || { echo "usage: $0 SPLIT RUN_ID [RUN_ID ...]" >&2; exit 1; }
SPLIT=$1
shift
RUN_IDS=$(IFS=,; echo "$*")
N=$#

HOST="${HOST:-fryderyk.koegl@hpc-submit02.scidom.de}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR="${SNAP_DIR:-/home/iml/fryderyk.koegl/code_snapshots}"  # on the cluster
SNAP="$SNAP_DIR/eval_$STAMP"

# snapshot the whole Code/ dir: modules are read when a task STARTS, so later
# edits must not reach queued tasks
rsync -a --exclude='__pycache__' --exclude='*.pyc' --exclude='.fuse_hidden*' \
    "$HERE/" "$HOST:$SNAP/"
git -C "$HERE" rev-parse HEAD | ssh "$HOST" "cat > $SNAP/.snapshot_git_head"
git -C "$HERE" diff HEAD | ssh "$HOST" "cat > $SNAP/.snapshot_git_diff"

JOBID=$(ssh "$HOST" "sbatch --parsable --array=0-$((N - 1)) \
    --export=ALL,CODE_DIR=$SNAP,RUN_IDS=$RUN_IDS,SPLIT=$SPLIT \
    $SNAP/psmareg_eval_array.sbatch")
ssh "$HOST" "echo $JOBID > $SNAP/.snapshot_jobid; \
    printf '%s\t%s\t%s\t%s\n' $JOBID $STAMP $SNAP '$RUN_IDS' >> $SNAP_DIR/index.tsv"
echo "Submitted array $JOBID ($SPLIT: $RUN_IDS) with snapshot $SNAP"
