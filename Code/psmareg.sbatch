#!/bin/bash
# Training job for the PSMAReg LapIRN model.
#
# Normally submitted through submit_with_config_snapshot.sh, which copies Code/
# to a timestamped snapshot and passes it as CODE_DIR, so edits made while the
# job waits in the queue cannot leak into the run:
#
#   bash Code/submit_with_config_snapshot.sh
#
# Submitting this file directly with `sbatch Code/psmareg.sbatch` also works and
# runs the live working tree -- fine for a quick test, but the run is then not
# reproducible from a snapshot.
#
# Lives in the repo rather than ~/jobs so it is versioned with the code it runs;
# sbatch copies the batch script at submit time, so the queue holds this exact
# content regardless of later edits.
#SBATCH --job-name=psmareg_full
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%j.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%j.err
#SBATCH -t 1-00:00:00
#SBATCH -p gpu_p
#SBATCH --mem=120G
#SBATCH --qos=gpu_normal
# H100 only. A bare `--gres=gpu:1` lets SLURM satisfy the request with whatever
# is free -- V100, A100 or H100 -- which makes training runs incomparable to
# each other and to the container timings, and can silently halve throughput.
# Naming the GRES *type* is what enforces it; the job stays queued until an H100
# node (supergpu[14-32]) frees up rather than starting on the wrong hardware.
#
# If the H100 nodes are contended and the run is not being compared against
# earlier ones, `--gres=gpu:a100_80gb:1` is the next best option.
#SBATCH --gres=gpu:h100:1

# pipefail only: `set -u` would break `source .venv/bin/activate`, whose
# activate script reads $PS1, and $CUDA_VISIBLE_DEVICES is legitimately unset on
# some allocations. Explicit checks below cover what actually matters.
set -o pipefail

# prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#export CUDA_LAUNCH_BLOCKING=1

# ── Environment ───────────────────────────────────────────────────────────────
# All overridable, so the job does not have to be edited to run from a different
# checkout or virtualenv. CODE_DIR is what submit_with_config_snapshot.sh sets
# to the snapshot; without it the live working tree is used.
REPO_DIR="${REPO_DIR:-/home/iml/fryderyk.koegl/code/LapIRN-koegl}"
CODE_DIR="${CODE_DIR:-$REPO_DIR/Code}"
VENV="${VENV:-/home/iml/fryderyk.koegl/code/endoKI/.venv}"
WANDB_ENV="${WANDB_ENV:-/home/iml/fryderyk.koegl/.wandb_env}"

# Fail here rather than 30 s into a job that would have died on an import error.
[[ -d "$CODE_DIR" ]] || { echo "no CODE_DIR: $CODE_DIR" >&2; exit 1; }
[[ -f "$VENV/bin/activate" ]] || { echo "no venv: $VENV" >&2; exit 1; }

cd "$REPO_DIR"
source "$VENV/bin/activate"
# Non-fatal: without it wandb logging degrades, but the training run is still
# worth having rather than dying at submit time over a missing key file.
[[ -f "$WANDB_ENV" ]] && source "$WANDB_ENV" || echo "WARNING: no $WANDB_ENV"

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started on $(hostname) at $(date)"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "code dir: $CODE_DIR"

# Print GPU info directly from nvidia-smi into the log
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -u "$CODE_DIR/Train_LapIRN_diff_overfit_one_PETCT.py"

echo "Job finished at $(date)"
