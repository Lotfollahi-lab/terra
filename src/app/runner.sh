#!/bin/bash
echo "NUM_NODES: ${NUM_NODES}"
echo "NUM_GPUS_NODE: ${NUM_GPUS_NODE}"
echo "TOTAL_NUM_GPUS: ${TOTAL_NUM_GPUS}"
echo "NUM_PROCESSES_NODE: ${NUM_PROCESSES_NODE}"
echo "TOTAL_NUM_CORES: ${TOTAL_NUM_CORES}"
echo "NUM_HOSTS: ${NUM_HOSTS}"
echo "GPU_PER_HOST: ${GPU_PER_HOST}"
echo "MASTER_ADDR is: ${MASTER_ADDR}"
echo "LSB_JOBID: ${LSB_JOBID}"
echo "MPI_HOST_STRING: ${MPI_HOST_STRING}"
echo "UCX_IB_MLX5_DEVX=${UCX_IB_MLX5_DEVX}"
echo "TRAINING_SCRIPT_PATH=${TRAINING_SCRIPT_PATH}"
echo "NUMBER_EPOCHS=${NUMBER_EPOCHS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "LOG_DIR=${LOG_DIR}"

mpirun \
    -np ${TOTAL_NUM_GPUS} \
    -H ${MPI_HOST_STRING} \
    -x PATH \
    -bind-to none \
    -map-by slot \
    --mca pml ob1 --mca btl ^openib \
    --display-allocation \
    --display-map \
    python3 ${TRAINING_SCRIPT_PATH} \
        --backend ${BACKEND} \
        --fname ${CONFIG_FILE} \