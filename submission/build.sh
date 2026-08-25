#!/usr/bin/env bash
# Build the submission image. Run from anywhere; the build context is the repo
# root, which is what lets the Dockerfile COPY Code/ without a copy of it here.
#
# The nnU-Net fork and its weights live outside this repo, so they come in as
# named build contexts instead of being duplicated. Override either with
# AUTOPET_DIR / NNUNET_MODEL_DIR.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-psmareg_lapirn}"

AUTOPET_DIR="${AUTOPET_DIR:-/home/iml/fryderyk.koegl/code/autopet-3-submission}"
NNUNET_MODEL_DIR="${NNUNET_MODEL_DIR:-/home/iml/fryderyk.koegl/data/PSMAReg-nnunet/nnUNet_results/Dataset501_PSMALesion/nnUNetTrainer_PGPSplus__nnUNetPlans__3d_fullres}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-$HOME/.totalsegmentator/nnunet/results}"

for d in "$AUTOPET_DIR" "$NNUNET_MODEL_DIR" "$TOTALSEG_WEIGHTS"; do
  [[ -d "$d" ]] || { echo "missing build context: $d" >&2; exit 1; }
done
[[ -f "$NNUNET_MODEL_DIR/fold_0/checkpoint_final.pth" ]] || {
  echo "no fold_0/checkpoint_final.pth in $NNUNET_MODEL_DIR" >&2; exit 1; }

cd "$REPO_ROOT"
DOCKER_BUILDKIT=1 docker build -f submission/Dockerfile -t "$IMAGE" \
  --build-context "autopet=$AUTOPET_DIR" \
  --build-context "nnunet_model=$NNUNET_MODEL_DIR" \
  --build-context "totalseg_weights=$TOTALSEG_WEIGHTS" \
  .
echo "built $IMAGE from $(git rev-parse --short HEAD)"
