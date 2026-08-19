#!/bin/bash
set -euo pipefail

# default code dir; override by passing a path as the first argument
SRC=$(realpath "${1:-/home/iml/fryderyk.koegl/code/LapIRN-koegl/Code}")
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR=/home/iml/fryderyk.koegl/code_snapshots
SNAP="$SNAP_DIR/$STAMP"

# Snapshot the whole Code/ dir, not just config.py: every module is read when
# the job actually starts, so edits made while the job waits in the queue would
# otherwise leak into the run.
mkdir -p "$SNAP"
tar -cf - -C "$SRC" \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.fuse_hidden*' . \
  | tar -xf - -C "$SNAP"

# record what the snapshot corresponds to
git -C "$SRC" rev-parse HEAD > "$SNAP/.snapshot_git_head" 2>/dev/null || true
git -C "$SRC" diff HEAD > "$SNAP/.snapshot_git_diff" 2>/dev/null || true

JOBID=$(sbatch --parsable --export=ALL,CODE_DIR="$SNAP" \
                /home/iml/fryderyk.koegl/jobs/psmareg.sh)

# record the job <-> snapshot link at submit time, so runs never have to be
# matched back to a snapshot by timestamp arithmetic
echo "$JOBID" > "$SNAP/.snapshot_jobid"
printf '%s\t%s\t%s\n' "$JOBID" "$STAMP" "$SNAP" >> "$SNAP_DIR/index.tsv"

echo "Submitted job $JOBID with snapshot: $SNAP"
