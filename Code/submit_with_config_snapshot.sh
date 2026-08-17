#!/bin/bash
set -euo pipefail
# default config; override by passing a path as the first argument
CONFIG=$(realpath "${1:-/home/iml/fryderyk.koegl/code/LapIRN-koegl/Code/config.py}")
STAMP=$(date +%Y%m%d_%H%M%S)
SNAP_DIR=/home/iml/fryderyk.koegl/config_snapshots
mkdir -p "$SNAP_DIR"
SNAP="$SNAP_DIR/${STAMP}_config.py"
cp "$CONFIG" "$SNAP"
sbatch --export=ALL,CFG="$SNAP" /home/iml/fryderyk.koegl/jobs/psmareg.sh
echo "Submitted with snapshot: $SNAP"