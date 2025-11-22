#!/bin/bash
#SBATCH --job-name=nichejepa_train
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --mem=480G
#SBATCH --constraint=h100_80gb
#SBATCH --time=48:00:00
#SBATCH --output=/home/aih/sebastian.birk/workspace/projects/nichejepa-reproducibility/logs/hst_corpus_70m/slurm_logs/%x_%j.out
#SBATCH --error=/home/aih/sebastian.birk/workspace/projects/nichejepa-reproducibility/logs/hst_corpus_70m/slurm_logs/%x_%j.err

# Set experiment identifiers (optional: used in your script)
export EXPERIMENT_NAME="hst_corpus_70m"
export RUN_NAME="gtsmall_subsample_combined_2"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "GPUs visible to this job: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load Python environment
export PATH="/home/aih/sebastian.birk/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nichejepa
echo "Using Python: $(which python)"
echo "Using Torchrun: $(which torchrun)"

srun torchrun \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
  /home/aih/sebastian.birk/workspace/projects/nichejepa/src/app/main_dist.py \
  --backend nccl \
  --fname /home/aih/sebastian.birk/workspace/projects/nichejepa/configs/model/hst_corpus_70m_gt_small.yaml