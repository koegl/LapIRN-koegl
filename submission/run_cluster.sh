#!/usr/bin/env bash
# Run the submission container over a validation set on a machine that is not
# the build host -- typically an HPC node with Apptainer instead of Docker.
#
# The image is identical to the one Docker runs; only the invocation differs.
# Apptainer has no --user (it already runs as you) and no --memory, so those
# flags are dropped rather than translated.
#
#   IMAGE=/path/psmareg.sif DATA_DIR=... OUTPUT_DIR=... bash run_cluster.sh
#
# Runtime is auto-detected; force it with RUNTIME=apptainer|singularity|docker.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_DIR="${DATA_DIR:-/lustre/groups/iml/data/PSMAReg/PSMAReg_dataset/imagesVal}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/predictions_cluster}"
DATASET_JSON="${DATASET_JSON:-$HERE/PSMAReg_val_dataset.json}"
LIMIT="${LIMIT:-0}"

if [[ -z "${RUNTIME:-}" ]]; then
  for r in apptainer singularity docker; do
    command -v "$r" >/dev/null 2>&1 && RUNTIME="$r" && break
  done
fi
: "${RUNTIME:?no apptainer/singularity/docker found; set RUNTIME=}"

if [[ "$RUNTIME" == "docker" ]]; then
  IMAGE="${IMAGE:-psmareg_lapirn}"
else
  IMAGE="${IMAGE:-$HERE/psmareg_lapirn.sif}"
  [[ -f "$IMAGE" ]] || { echo "no image at $IMAGE (set IMAGE=)" >&2; exit 1; }
fi

[[ -d "$DATA_DIR" ]] || { echo "no data dir: $DATA_DIR" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

mapfile -t SUBJECTS < <(python3 -c "
import json
for e in json.load(open('$DATASET_JSON'))['validation_paired']:
    print(e['subject'].split('_')[-1])
")
if [[ "$LIMIT" -gt 0 ]]; then SUBJECTS=("${SUBJECTS[@]:0:$LIMIT}"); fi

# Fail loudly here rather than producing 20 identical warnings: a wrong DATA_DIR
# is the most likely difference between this machine and the build host.
missing=()
for id in "${SUBJECTS[@]}"; do
  [[ -f "$DATA_DIR/PSMARegPSMA_${id}_0000_00.nii.gz" ]] || missing+=("$id")
done
if (( ${#missing[@]} )); then
  echo "missing inputs for: ${missing[*]}" >&2
  echo "check DATA_DIR=$DATA_DIR" >&2
  exit 1
fi

echo "runtime=$RUNTIME image=$IMAGE"
echo "${#SUBJECTS[@]} pairs: $DATA_DIR -> $OUTPUT_DIR"

# DEV=1 mounts the working tree over the image's baked copy, the same trick
# test.sh uses. On the cluster it matters much more: the alternative to changing
# three Python files is rebuilding the image on the build host, `docker save`ing
# several GB, transferring it, and converting it to a SIF. Nothing here changes
# the environment, only the method code -- so it is valid for timing and scoring
# runs, but the SUBMITTED image must be a real rebuild, never a bound tree.
REPO_ROOT="$(cd "$HERE/.." && pwd)"
DEV_DOCKER=()
DEV_APPTAINER=()
if [[ "${DEV:-0}" == "1" ]]; then
  for pair in \
    "$REPO_ROOT/Code:/app/lapirn/Code" \
    "$REPO_ROOT/submission/infer.py:/app/infer.py" \
    "$REPO_ROOT/submission/totalseg_runner.py:/app/totalseg_runner.py" \
    "$REPO_ROOT/time_totalsegmentator.py:/app/lapirn/time_totalsegmentator.py"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    [[ -e "$src" ]] || { echo "DEV=1 but missing $src" >&2; exit 1; }
    DEV_DOCKER+=(--mount "type=bind,source=$src,target=$dst,readonly")
    DEV_APPTAINER+=(--bind "$src:$dst:ro")
  done
  echo "dev mode: Code/, infer.py and the TotalSegmentator runner bind-mounted"
fi

run_one() {
  local id="$1"
  local args=(
    "/app/input/PSMARegPSMA_${id}_0000_00.nii.gz"
    "/app/input/PSMARegPSMA_${id}_0001_00.nii.gz"
    "/app/input/PSMARegPSMA_${id}_0000_01.nii.gz"
    "/app/input/PSMARegPSMA_${id}_0001_01.nii.gz"
    "/app/output/disp_${id}_00_${id}_01.nii.gz"
    --seg-dir /app/output/segmentations
    --ct-seg-dir /app/output/ct_labels
  )
  if [[ "$RUNTIME" == "docker" ]]; then
    docker run --rm --ipc=host --memory 60g \
      ${GPU_ARGS:---gpus device=0} \
      --user "$(id -u):$(id -g)" --network=none \
      --mount "type=bind,source=$DATA_DIR,target=/app/input,readonly" \
      --mount "type=bind,source=$OUTPUT_DIR,target=/app/output" \
      "${DEV_DOCKER[@]}" \
      "$IMAGE" "${args[@]}"
  else
    # --cleanenv: without it the host environment leaks in and overrides the
    #   image's HOME/TORCH_HOME/PYTHONPATH, which the container relies on.
    # --nv: bind the host NVIDIA driver, the equivalent of --gpus.
    "$RUNTIME" run --nv --cleanenv \
      --bind "$DATA_DIR:/app/input:ro" \
      --bind "$OUTPUT_DIR:/app/output" \
      "${DEV_APPTAINER[@]}" \
      "$IMAGE" "${args[@]}"
  fi
}

SWEEP_START=$SECONDS
TOTAL=0
N_PAIRS=${#SUBJECTS[@]}
LOG="$(mktemp)"
trap 'rm -f "$LOG"' EXIT
for id in "${SUBJECTS[@]}"; do
  CASE_START=$SECONDS
  # Same accounting as test.sh: the runtime's start-up before exec and the
  # teardown after the process exits are invisible from inside the container,
  # but both are inside the challenge's per-pair budget. Apptainer's start-up
  # is not Docker's, so this has to be measured on whichever runtime is used.
  RUN_START=$(date +%s.%N)
  run_one "$id" 2>&1 | tee "$LOG"
  RUN_END=$(date +%s.%N)
  echo "  $id: $((SECONDS - CASE_START))s (wall, incl. container startup)"
  EXEC_EPOCH=$(sed -n 's/.*exec at epoch \([0-9.]*\).*/\1/p' "$LOG" | tail -1)
  IN_CONTAINER=$(sed -n 's/.*in-container total \([0-9.]*\)s.*/\1/p' "$LOG" | tail -1)
  if [[ -n "$EXEC_EPOCH" && -n "$IN_CONTAINER" ]]; then
    awk -v s="$RUN_START" -v e="$RUN_END" -v x="$EXEC_EPOCH" -v c="$IN_CONTAINER" \
      'BEGIN { printf "    wall %.2fs = runtime start-up %.2fs + in-container %.2fs + teardown %.2fs\n", \
               e - s, x - s, c, (e - s) - (x - s) - c }'
  fi
done

TOTAL=$((SECONDS - SWEEP_START))
if (( N_PAIRS > 0 )); then
  echo "total ${TOTAL}s for ${N_PAIRS} pairs -> $((TOTAL / N_PAIRS))s/pair"
  echo "extrapolated to 200 test pairs: $((TOTAL * 200 / N_PAIRS / 60)) min (budget: 300 min)"
fi
