#!/bin/bash
#!/bin/bash
#SBATCH --job-name=nemo-test             # sets the name of the job shown in the job queue (via `squeue`)
#SBATCH --nodes=4                        # requests 2 nodes (each will typically run one process)
#SBATCH --ntasks-per-node=1              # runs one task (process) per node and aligns with DDP across nodes
#SBATCH --gpus-per-node=1                # requests 1 GPU on each node
#SBATCH --cpus-per-task=1                # allocates 1 CPU cores per task
#SBATCH --mem=16G                        # allocates 16 GB of RAM memory per node
#SBATCH --partition=g6e-8xlarge          # specifies the partition (queue) to submit the job to (use `sinfo` to see avaialble)
#SBATCH --time=00:30:00                  # sets the max wall time (runtime) for the job (HH:MM:SS)
#SBATCH --output=logs/%j.out             # stdout file (%j is replaced with the job ID)
#SBATCH --error=logs/%j.err              # stder file (for logging errors)
#SBATCH --chdir=/home/ubuntu/nichejepa-reproducibility        # set working directory

# load modules
module load libfabric-aws/2.1.0amzn2.0
module load openmpi5/5.0.6

# configure NCCL for aws
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep -vE 'lo|docker')
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set experiment identifiers (optional: used in your script)
export EXPERIMENT_NAME="hst_corpus_80m"
export RUN_NAME="gtsmall_aws_test_1"

export RDZV_HOST=$(hostname)
export RDZV_PORT=29400

echo "SLURM_JOB_GPUS: $SLURM_JOB_GPUS"
echo "GPUs visible to this job: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# Load Python environment
source ../nichejepa_env/bin/activate
echo "Using Python: $(which python)"
echo "Using Torchrun: $(which torchrun)"

srun torchrun \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
  /home/ubuntu/nichejepa/src/app/main_dist.py \
  --backend nccl \
  --fname /home/ubuntu/nichejepa/configs/model/hst_corpus_80m/hst_corpus_80m_gtsmall_aws.yaml
