#!/bin/bash
# Run the LION PSMA model (docker) on the prepared subject folders.
# Snap docker: GPUs via --runtime=nvidia (--gpus is not supported).
# The dataset folder is mounted at its host path so the PT_ symlinks resolve inside the container.
# lionz 1.0.5: the 1.0.6 tag from the README is not on Docker Hub
set -euo pipefail
WORK=${1:-/home/iml/fryderyk.koegl/data/PSMAReg/lion_eval}
DATA=/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=2g -e LIONZ_TELEMETRY=0 \
  -v "$WORK":/shared \
  -v "$DATA":"$DATA":ro \
  -v lionz-models:/usr/local/models \
  lalithshiyam/lionz:1.0.5 \
  -d /shared -m psma
