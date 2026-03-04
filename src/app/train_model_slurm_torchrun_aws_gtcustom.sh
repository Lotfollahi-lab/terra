#!/bin/bash
#SBATCH --job-name=nemo-pretraining      # sets the name of the job shown in the job queue (via `squeue`)
#SBATCH --nodes=4                        # requests 4 nodes (each will typically run one process)
#SBATCH --ntasks-per-node=1              # runs one task (process) per node and aligns with DDP across nodes
#SBATCH --gpus-per-node=8                # requests 8 GPU on each node
#SBATCH --cpus-per-task=48               # allocates 48 CPU cores per task
#SBATCH --mem=1024G                      # allocates 1024 GB of RAM memory per node
#SBATCH --partition=p4de-24xlarge        # specifies the partition (queue) to submit the job to (use `sinfo` to see avaialble)
#SBATCH --time=480:00:00                 # sets the max wall time (runtime) for the job (HH:MM:SS)
#SBATCH --output=logs/aws/%j.out         # stdout file (%j is replaced with the job ID)
#SBATCH --error=logs/aws/%j.err          # stder file (for logging errors)
#SBATCH --chdir=/home/ubuntu/sb75/nichejepa-reproducibility        # set working directory


###############################################################################
# run this script as: `sbatch train_model_slurm_torchrun_aws.sh`
###############################################################################

set -oe pipefail
mkdir -p logs

# load modules
module load libfabric-aws/2.1.0amzn2.0
module load openmpi5/5.0.6

# configure NCCL for aws
export NCCL_SOCKET_IFNAME='^lo,docker'
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# configure log level
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

export TMPDIR=/fsx-shared/tmp

# activate environment
source ../nichejepa_env/bin/activate

# Set master address and port
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$((12000 + RANDOM % 1000)) 

export EXPERIMENT_NAME="hst_corpus_90m" # "hst_corpus_80m"
export RUN_NAME="gtcustom_fullcorpus_aws_1"

echo "[+] SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "[+] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "[+] SLURM_LOCALID: $SLURM_LOCALID"
echo "[+] SLURM_NODEID: $SLURM_NODEID"
nvidia-smi

echo "[+] python: $(which python)"
echo "[+] torchrun: $(which torchrun)"
echo "    --nproc_per_node $SLURM_GPUS_PER_NODE"
echo "    --nnodes $SLURM_NNODES"
echo "    --node_rank $SLURM_NODEID"
echo "    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT"
echo "    --rdzv_backend c10d"

# Run with torchrun
srun torchrun \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_NODEID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    /home/ubuntu/sb75/nichejepa/src/app/main_dist.py \
    --backend nccl \
    --fname /home/ubuntu/sb75/nichejepa/configs/model/hst_corpus_90m/hst_corpus_90m_gtcustom_aws.yaml