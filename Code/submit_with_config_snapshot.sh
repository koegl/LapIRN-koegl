#!/bin/bash
set -euo pipefail

# Usage: submit_with_config_snapshot.sh {iml|general} [SRC_DIR]
#   iml     -> psmareg_iml.sbatch     (IML reservation, 4 days, max 2 parallel)
#   general -> psmareg_general.sbatch (general pool, 1 day, H100/A100-80GB)
usage() { echo "usage: $0 {iml|general} [SRC_DIR]" >&2; exit 1; }
[[ $# -ge 1 ]] || usage
TARGET=$1
[[ "$TARGET" == iml || "$TARGET" == general ]] || usage

# This script lives in Code/, so the directory to snapshot is its own -- no
# absolute path needed, and a second checkout works without editing anything.
# Override by passing a path as the second argument.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(realpath "${2:-$HERE}")
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR="${SNAP_DIR:-/home/iml/fryderyk.koegl/code_snapshots}"
SNAP="$SNAP_DIR/$STAMP"

# The batch script is versioned with the code now (Code/psmareg_<target>.sbatch),
# not kept in ~/jobs. The repo copy is submitted rather than the snapshot's: sbatch
# takes its own copy of the batch script at submit time, so there is no window
# in which a later edit could reach a queued job. The snapshot exists for the
# Python modules, which are read when the job STARTS.
JOB_SCRIPT="${JOB_SCRIPT:-$HERE/psmareg_${TARGET}.sbatch}"
[[ -f "$JOB_SCRIPT" ]] || { echo "no job script: $JOB_SCRIPT" >&2; exit 1; }

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

JOBID=$(sbatch --parsable --export=ALL,CODE_DIR="$SNAP" "$JOB_SCRIPT")

# record the job <-> snapshot link at submit time, so runs never have to be
# matched back to a snapshot by timestamp arithmetic
echo "$JOBID" > "$SNAP/.snapshot_jobid"
printf '%s\t%s\t%s\n' "$JOBID" "$STAMP" "$SNAP" >> "$SNAP_DIR/index.tsv"

echo "Submitted job $JOBID ($TARGET) with snapshot: $SNAP"
