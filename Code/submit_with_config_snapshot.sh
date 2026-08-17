#!/bin/bash
set -euo pipefail
CONFIG=$(realpath "$1")
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR=/home/iml/fryderyk.koegl/config_snapshots
mkdir -p "$SNAP_DIR"
SNAP="$SNAP_DIR/${STAMP}_config.py"
cp "$CONFIG" "$SNAP"
sbatch --export=ALL,CFG="$SNAP" /home/iml/fryderyk.koegl/jobs/psmareg.sh
echo "Submitted with snapshot: $SNAP"