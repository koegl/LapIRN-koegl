#!/bin/bash
# Run the submission container over the validation set on the cluster.
#
#   sbatch submission/submit_cluster.sbatch
#
# Build the .sif once first (see below); this script only runs it.
#SBATCH --job-name=psmareg_container
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%j.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%j.err
#SBATCH -t 04:00:00
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
# H100 only: supergpu[14-32] expose GRES type `h100` (feature `h100_80gb`).
# Without the type SLURM may hand out a V100 or A100 and the timings would not
# describe the hardware the challenge is evaluated on.
#SBATCH --gres=gpu:h100:1
# 60G and 12 CPUs mirror the challenge evaluation environment (Xeon Silver
# 4410Y, 60 GB) rather than what this cluster could give us. The point of this
# run is to find out whether the container fits THEIR box, so asking for more
# would hide exactly the failure we are looking for.
#SBATCH --mem=60G
#SBATCH --cpus-per-task=12
#
# If the H100 nodes are contended, `--gres=gpu:a100_80gb:1` is the next best
# portability test -- the speed numbers change but driver and CUDA behaviour
# still transfer.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/iml/fryderyk.koegl/code/LapIRN-koegl}"
SIF="${SIF:-$HOME/containers/psmareg_lapirn.sif}"
DATA_DIR="${DATA_DIR:-/lustre/groups/iml/data/PSMAReg/PSMAReg_dataset/imagesVal}"
# ~2.5 GB of displacement fields plus segmentations for 20 pairs -- override with
# OUTPUT_DIR if $HOME is quota-limited on this cluster.
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/psmareg_pred_cluster}"
LIMIT="${LIMIT:-0}"

echo "=================== job ==================="
echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "=================== cpu ==================="
# The container's registration stage is ~20 s of single-threaded CPU (ANTs plus
# the body masks), so the CPU model matters as much as the GPU for the runtime
# budget -- and it is the part least likely to match the evaluation machine.
lscpu | grep -E "^Model name|^CPU\(s\)|^Thread|^Core|MHz" || true
echo "allocated cpus: ${SLURM_CPUS_PER_TASK:-?}   mem: ${SLURM_MEM_PER_NODE:-?} MB"
echo "=================== gpu ==================="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=============== container ================="
(apptainer --version || singularity --version) 2>/dev/null || echo "no apptainer/singularity"
echo "sif: $SIF"
echo "==========================================="

mkdir -p "$OUTPUT_DIR"

IMAGE="$SIF" DATA_DIR="$DATA_DIR" OUTPUT_DIR="$OUTPUT_DIR" LIMIT="$LIMIT" \
    bash "$REPO_DIR/submission/run_cluster.sh"

echo "job finished at $(date)"
