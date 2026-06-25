#!/bin/bash

# Load site paths (TERRA_*) from the git-ignored cluster_env.sh so the
# ${...} variables used in the job config resolve at submit time.
_TERRA_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_TERRA_SCRIPTS_DIR/cluster_env.sh" ]; then
    . "$_TERRA_SCRIPTS_DIR/cluster_env.sh"
fi

set -euo pipefail

export ADVANCED_BASH_SCRIPT_PATH="/software/isg/advanced-bsub-script"

# Source the check_python.sh script to access the check_python_version function
source ${ADVANCED_BASH_SCRIPT_PATH}/check_python.sh  # Update this path to the actual location of check_python.sh

# Call the function to check Python version
check_python_version

CONFIG_FILE="$1"

eval $(python3 ${ADVANCED_BASH_SCRIPT_PATH}/parse_config.py "$CONFIG_FILE")

echo "QUEUE=$QUEUE"                 # Queue name
echo "NUM_NODES=$NUM_NODES"         # Number of nodes
echo "NUM_GPUS_NODE=$NUM_GPUS_NODE"         # Number of GPUs per node
echo "NUM_PROCESSES_NODE=$NUM_PROCESSES_NODE"    # Number of processes per node
echo "MEM_NODE=$MEM_NODE"              # Memory per node in GB
echo "ENV_PATH=$ENV_PATH"              # Environment activation path
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"       # 0 using InfiniBand, 1 using TCP
# echo "CUDA_VERSION=$CUDA_VERSION"  # CUDA version
echo "WAREHOUSE_PATH=$WAREHOUSE_PATH"     # Artifact location (default empty)
echo "EXPERIMENT_NAME=$EXPERIMENT_NAME"       # Experiment name (default empty)
echo "RUN_NAME=$RUN_NAME"              # Run name (default empty)
echo "RUNNER_SCRIPT=$RUNNER_SCRIPT"         # Script name (default empty)
echo "OUTPUT_DIR=$OUTPUT_DIR"               # Output directory (default empty)
echo "LOG_DIR=$LOG_DIR"               # Log directory (default empty)

export CPROFILE_FILE_NAME="${OUTPUT_DIR}/${RUN_NAME}.prof"

# Set PYTHONPYCACHEPREFIX to the experiment directory
#export PYTHONPYCACHEPREFIX="/tmp/pycache"

# Calculating Total Number of GPUs
TOTAL_NUM_GPUS=$(($NUM_NODES * $NUM_GPUS_NODE))

export TOTAL_NUM_GPUS=$TOTAL_NUM_GPUS

# Arithmetic evaluation for TOTAL_NUM_CORES
TOTAL_NUM_CORES=$(($NUM_NODES * $NUM_PROCESSES_NODE))

export TOTAL_NUM_CORES=$TOTAL_NUM_CORES

# Export environment variables for NCCL and MPI
export NCCL_DEBUG=INFO
export NCCL_TOPO_DUMP_FILE="${LOG_DIR}/nccl_topo_dump-%J.xml"
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_debug-%J.log"
# Disable NVLink-multicast (NVLS) collectives. NCCL >=2.2x enables NVLS by
# default on NVSwitch GPUs, but the cuMulticast* setup fails on this node's
# driver/fabric with "transport/nvls.cc Cuda failure 1 'invalid argument'".
# Falling back to ring/tree collectives is correct (marginally slower
# allreduce on NVSwitch nodes). Remove if the driver/NCCL combo is fixed.
export NCCL_NVLS_ENABLE=0

# Check if NCCL_IB_DISABLE is set
. "${ADVANCED_BASH_SCRIPT_PATH}/nccl_setup.sh"

#----
export TORCH_CPP_LOG_LEVEL="INFO"
#----

# Submit the job using bsub
bsub <<EOF
#BSUB -J ${EXPERIMENT_NAME}_${RUN_NAME}
#BSUB -o ${LOG_DIR}/${EXPERIMENT_NAME}_${RUN_NAME}_o.%J
#BSUB -e ${LOG_DIR}/${EXPERIMENT_NAME}_${RUN_NAME}_e.%J
#BSUB -n ${TOTAL_NUM_CORES}
#BSUB -q "${QUEUE}"
#BSUB -G s10396
#BSUB -U lotfollahi-training-normal
#BSUB -gpu "num=$NUM_GPUS_NODE:gmem=80000:mode=exclusive_process:block=yes"
#BSUB -M ${MEM_NODE}G
#BSUB -R "select[mem>${MEM_NODE}G] rusage[mem=${MEM_NODE}G] span[ptile=$NUM_PROCESSES_NODE]"

# Set the error handling mode to pipefail
set -euo pipefail

export NCCL_TOPO_DUMP_FILE="${LOG_DIR}/nccl_topo_dump-\${LSB_JOBID}.xml"
export NCCL_DEBUG_FILE="${LOG_DIR}/nccl_debug-\${LSB_JOBID}.log"

# Unset the variable causing the error
unset LSB_AFFINITY_HOSTFILE

# Force initialisation of module system if you are doing non standard stuff
. /usr/share/modules/init/sh
module load ISG/experimental/fg12/openmpi/5.0.4-cuda12.1-lsf

# Load CUDA
source ${ADVANCED_BASH_SCRIPT_PATH}/check_and_load_cuda.sh
check_and_load_cuda

# Set app environment
source ${ENV_PATH}/bin/activate

which python3
python3 --version

# Find the master port
. "${ADVANCED_BASH_SCRIPT_PATH}/find_master_port.sh"

# Build the -H argument for mpirun
if [ -n "\${DEEPSPEED_HOSTFILE:-}" ]; then
    . "${ADVANCED_BASH_SCRIPT_PATH}/make_hostfile.sh"
else
    . "${ADVANCED_BASH_SCRIPT_PATH}/build_mpi_host_string.sh"
fi

echo "##----------------------------------------------------"
# Check if ulimit is increased
ulimit -a
echo "##----------------------------------------------------"

${RUNNER_SCRIPT}
EOF
