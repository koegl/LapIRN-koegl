#!/bin/bash
set -euo pipefail

# Usage (on janus): submit_eval_array.sh {iml|general} SPLIT RUN_ID [RUN_ID ...]
# Copies Code/ to a timestamped snapshot on the cluster and submits
# psmareg_eval_array.sbatch as an array with one evaluation run per task.
#   iml     -> IML reservation (short queue, at most 2 tasks at once)
#   general -> general pool (all tasks in parallel, longer queue)
# Results: $PSMAREG_DATA_PATH/evaluation/<split>/ on the cluster.
usage() { echo "usage: $0 {iml|general} SPLIT RUN_ID [RUN_ID ...]" >&2; exit 1; }
[[ $# -ge 3 ]] || usage
TARGET=$1
SPLIT=$2
shift 2
case "$TARGET" in
    iml) QUEUE_ARGS="--qos=gpu_reservation --reservation=iml_user"; THROTTLE="%2" ;;
    general) QUEUE_ARGS=""; THROTTLE="" ;;
    *) usage ;;
esac
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

JOBID=$(ssh "$HOST" "sbatch --parsable --array=0-$((N - 1))$THROTTLE $QUEUE_ARGS \
    --export=ALL,CODE_DIR=$SNAP,RUN_IDS=$RUN_IDS,SPLIT=$SPLIT \
    $SNAP/psmareg_eval_array.sbatch")
ssh "$HOST" "echo $JOBID > $SNAP/.snapshot_jobid; \
    printf '%s\t%s\t%s\t%s\n' $JOBID $STAMP $SNAP '$RUN_IDS' >> $SNAP_DIR/index.tsv"
echo "Submitted array $JOBID ($TARGET, $SPLIT: $RUN_IDS) with snapshot $SNAP"
