#!/bin/bash
# Evaluation runs (evaluate.py) as a Slurm array: one evaluation run ID per
# array task, so independent IO runs execute in parallel.
#
# Submitted from janus through submit_eval_array.sh, which copies Code/ to a
# snapshot on the cluster and passes CODE_DIR, RUN_IDS (comma-separated, one per
# array index) and SPLIT:
#
#   bash Code/submit_eval_array.sh test W0 W0.0001 G1
#
# Data: a self-contained copy of the evaluation split (images, GT and predicted
# labels, janus's affine cache, journal checkpoints, split file, tuned IO
# settings) in $PSMAREG_DATA_PATH, laid out like DATA_PATH on janus, so the
# results are comparable with the runs evaluated on janus. Results go to
# $PSMAREG_DATA_PATH/evaluation/<split>/ (sync back with rsync).
#SBATCH --job-name=psmareg_eval
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%A_%a.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%A_%a.err
# one run (38 test pairs of IO) takes ~2 h on janus's A100; a short limit
# lets the scheduler backfill the tasks into gaps
#SBATCH -t 04:00:00
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --constraint="h100_80gb|a100_80gb"

set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PSMAREG_DATA_PATH="${PSMAREG_DATA_PATH:-/lustre/groups/iml/data/PET_CT_bone/test_set_for_eval}"
VENV="${VENV:-/home/iml/fryderyk.koegl/code/endoKI/.venv}"
SPLIT="${SPLIT:-test}"

[[ -d "$CODE_DIR" ]] || { echo "no CODE_DIR: $CODE_DIR" >&2; exit 1; }
[[ -n "$RUN_IDS" ]] || { echo "no RUN_IDS" >&2; exit 1; }
[[ -d "$PSMAREG_DATA_PATH/PSMAReg" ]] || { echo "no data: $PSMAREG_DATA_PATH" >&2; exit 1; }
IFS=',' read -ra RUNS <<< "$RUN_IDS"
RUN_ID="${RUNS[$SLURM_ARRAY_TASK_ID]}"
[[ -n "$RUN_ID" ]] || { echo "no run for array index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

source "$VENV/bin/activate"
cd "$CODE_DIR"
echo "Job started on $(hostname) at $(date): run $RUN_ID ($SPLIT), code $CODE_DIR"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -u - "$RUN_ID" "$SPLIT" <<'EOF'
import os
import sys
from pathlib import Path

import config

data = Path(os.environ["PSMAREG_DATA_PATH"])
config.TrainingConfig.split_path_journal = data / "config" / "split_journal.json"

import evaluate

run_id, split = sys.argv[1], sys.argv[2]
evaluate.SPLIT = split
evaluate.RUN_IDS = [run_id]
evaluate.OUT_DIR = data / "evaluation"
evaluate.IO_TUNED_PATH = data / "config" / "io_tuned.json"
evaluate.main()
EOF

echo "Job finished at $(date)"
