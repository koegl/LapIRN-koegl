#!/usr/bin/env bash
# Run the container once per validation pair, exactly the way the organizers
# invoke it (§4 of the challenge instructions), and time the whole sweep.
#
#   DEV=1 bash test.sh    bind-mount Code/ instead of using the baked copy,
#                         so code changes take effect without a rebuild
#   LIMIT=1 bash test.sh   only the first pair (quick timing check)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-psmareg_lapirn}"
DATA_DIR="${DATA_DIR:-/home/iml/fryderyk.koegl/data/PSMAReg/PSMAReg_dataset/imagesTs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/submission/validation_predictions}"
DATASET_JSON="${DATASET_JSON:-$REPO_ROOT/submission/PSMAReg_val_dataset.json}"
LIMIT="${LIMIT:-0}"
# PET lesion masks are written next to the fields so evaluate_disp.py can score
# them. SEG=0 turns the segmentation off entirely (registration-only timing).
EXTRA_ARGS=(--seg-dir /app/output/segmentations --ct-seg-dir /app/output/ct_labels)
if [[ "${SEG:-1}" == "0" ]]; then
  EXTRA_ARGS=(--no-segment --no-io)
fi
# The axial band is cfg.io_seg_z_range in Code/config.py, with io_it and io_lr --
# bind-mounted under DEV=1, so sweeping it needs no rebuild. TOTALSEG=0 skips
# the CT labels entirely.
if [[ "${TOTALSEG:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-totalseg)
fi
# IO step count and learning rate come from Code/config.py (io_it / io_lr),
# alongside the w_io_* weights -- with DEV=1 that file is bind-mounted, so
# retuning them needs no rebuild. IO=0 turns the stage off for timing runs.
if [[ "${IO:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-io)
fi

# GPU flags. The organizers use `--gpus "device=0"`; this workstation's snap
# Docker runs the NVIDIA hook in CDI mode, which rejects --gpus and demands the
# runtime be selected explicitly. Both request the same device -- this only
# affects local testing, never the submitted image.
#   GPU_ARGS='--gpus device=0' bash test.sh   to use the organizers' exact form
read -r -a GPU_ARGS <<< "${GPU_ARGS:---runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=compute,utility}"

mkdir -p "$OUTPUT_DIR"

mapfile -t SUBJECTS < <(python3 -c "
import json, sys
entries = json.load(open('$DATASET_JSON'))['validation_paired']
for e in entries:
    print(e['subject'].split('_')[-1])
")
if [[ "$LIMIT" -gt 0 ]]; then
  SUBJECTS=("${SUBJECTS[@]:0:$LIMIT}")
fi

DEV_MOUNT=()
if [[ "${DEV:-0}" == "1" ]]; then
  # infer.py is mounted too: the crop and the stage toggles live there, so a
  # sweep would otherwise need a rebuild per value.
  DEV_MOUNT=(
    --mount "type=bind,source=$REPO_ROOT/Code,target=/app/lapirn/Code,readonly"
    --mount "type=bind,source=$REPO_ROOT/submission/infer.py,target=/app/infer.py,readonly"
    --mount "type=bind,source=$REPO_ROOT/submission/totalseg_runner.py,target=/app/totalseg_runner.py,readonly"
    --mount "type=bind,source=$REPO_ROOT/time_totalsegmentator.py,target=/app/lapirn/time_totalsegmentator.py,readonly"
  )
  echo "dev mode: Code/, infer.py and the TotalSegmentator runner bind-mounted"
fi

echo "${#SUBJECTS[@]} pairs -> $OUTPUT_DIR"
SWEEP_START=$SECONDS
# pre-declared: the summary below must still print if anything odd happens to
# the loop, since it is the only place the per-pair mean is reported
TOTAL=0
N_PAIRS=${#SUBJECTS[@]}

for id in "${SUBJECTS[@]}"; do
  CASE_START=$SECONDS
  docker run --rm \
    --ipc=host \
    --memory 60g \
    "${GPU_ARGS[@]}" \
    --user "$(id -u):$(id -g)" \
    --network=none \
    --mount "type=bind,source=$DATA_DIR,target=/app/input,readonly" \
    --mount "type=bind,source=$OUTPUT_DIR,target=/app/output" \
    "${DEV_MOUNT[@]}" \
    "$IMAGE" \
      "/app/input/PSMARegPSMA_${id}_0000_00.nii.gz" \
      "/app/input/PSMARegPSMA_${id}_0001_00.nii.gz" \
      "/app/input/PSMARegPSMA_${id}_0000_01.nii.gz" \
      "/app/input/PSMARegPSMA_${id}_0001_01.nii.gz" \
      "/app/output/disp_${id}_00_${id}_01.nii.gz" \
      "${EXTRA_ARGS[@]}"
  echo "  $id: $((SECONDS - CASE_START))s (wall, incl. container startup)"
done

TOTAL=$((SECONDS - SWEEP_START))
if (( N_PAIRS > 0 )); then
  echo "total ${TOTAL}s for ${N_PAIRS} pairs -> $((TOTAL / N_PAIRS))s/pair"
  echo "extrapolated to 200 test pairs: $((TOTAL * 200 / N_PAIRS / 60)) min (budget: 300 min)"
fi
