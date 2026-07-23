#!/bin/bash
#SBATCH --job-name=ribseg
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%A_%a.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%A_%a.err
#SBATCH -t 3-00:00:00
#SBATCH -p gpu_p
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --qos=gpu_reservation
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --reservation=iml_user
#SBATCH --array=0-4

NUM_SHARDS=5

# prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Environment ───────────────────────────────────────────────────────────────
cd /home/iml/fryderyk.koegl/code/
source /home/iml/fryderyk.koegl/code/endoKI/.venv/bin/activate

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started on $(hostname) at $(date)"
echo "Array task $SLURM_ARRAY_TASK_ID of $NUM_SHARDS shards"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Print GPU info directly from nvidia-smi into the log
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python -u /home/iml/fryderyk.koegl/code/LapIRN-koegl/segment_ribs_sbatch.py \
    --shard "$SLURM_ARRAY_TASK_ID" \
    --num-shards "$NUM_SHARDS"

echo "Job finished at $(date)"
