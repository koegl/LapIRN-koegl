#!/bin/bash
# One stage of the official-scorer re-evaluation on the cluster: array task i
# runs task i of the stage (tasks.STAGES), forward; janus works backward
# (worker.py janus). Claims on lustre keep the two apart.
#
# Submitted from janus with submit_stage.sh, which copies Code/ and
# official_scorer/ to a snapshot on the cluster and passes CODE_ROOT and STAGE.
#SBATCH --job-name=psmareg_official
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%A_%a.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%A_%a.err
# one task (38 pairs with IO) takes ~45 min on an H100
#SBATCH -t 04:00:00
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
# IO peaks at ~33 GB of GPU memory
#SBATCH --constraint="h100_80gb|a100_80gb|a100_40gb"

set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PSMAREG_DATA_PATH="${PSMAREG_DATA_PATH:-/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval}"
# temporary files on lustre, not in the node's /tmp (a full /tmp on one node
# killed tasks during the imports); no wandb logging
export TMPDIR="$PSMAREG_DATA_PATH/tmp/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
export WANDB_MODE=disabled WANDB_DIR="$TMPDIR"
VENV="${VENV:-/home/iml/fryderyk.koegl/code/endoKI/.venv}"

[[ -d "$CODE_ROOT/Code" && -d "$CODE_ROOT/official_scorer" ]] || { echo "bad CODE_ROOT: $CODE_ROOT" >&2; exit 1; }
[[ -n "$STAGE" ]] || { echo "no STAGE" >&2; exit 1; }

source "$VENV/bin/activate"
cd "$CODE_ROOT/Code"
echo "Job started on $(hostname) at $(date): stage $STAGE, task $SLURM_ARRAY_TASK_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -u "$CODE_ROOT/official_scorer/worker.py" cluster "$STAGE" "$SLURM_ARRAY_TASK_ID"
status=$?
rm -rf "$TMPDIR"
echo "Job finished at $(date) (exit $status)"
exit $status
