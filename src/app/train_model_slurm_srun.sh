#!/bin/bash
#SBATCH --job-name=nichejepa_train
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_long
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=630G
#SBATCH --constraint=h100_80gb
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Set experiment identifiers (optional: used in your script)
export EXPERIMENT_NAME="hst_corpus_70m"
export RUN_NAME="gtsmall_subsample_combined_1"

export MASTER_PORT=$((12000 + RANDOM % 1000))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR

# Load Python environment
export PATH="/home/aih/sebastian.birk/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate nichejepa
echo "Using Python: $(which python)"
echo "Using Torchrun: $(which torchrun)"

srun python \
  /home/aih/sebastian.birk/workspace/projects/nichejepa/src/app/main_dist.py \
  --backend nccl \
  --fname /home/aih/sebastian.birk/workspace/projects/nichejepa/configs/model/hst_corpus_70m_gt_small.yaml