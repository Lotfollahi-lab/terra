#!/bin/bash

GROUP="team361"

# Set the experiment parameters
EXPERIMENT_NAME="${1:-hst_corpus_70m}"
GT_TYPE="${2:-gt-base}"
DEVICES="${3:-cuda:0}"
RUN_ID="${4:-test-1}"
LOG_DIR="logs/${EXPERIMENT_NAME}/farm-job-outputs/model"

SCRIPT="/lustre/scratch126/cellgen/team361/sb75/nichejepa/src/app/train_model.sh"

case $EXPERIMENT_NAME in
    hst_corpus_70m)
        RAM="448G"
        TIME="240:00"
        CORES=128
        QUEUE="gpu-cellgen-restricted"
        N_GPU=8
        ;;
    *)
        echo "Invalid dataset name"
        exit 1
        ;;
esac

# If not provided, use the default value
CORES="${5:-$CORES}"
QUEUE="${6:-$QUEUE}"

# Set the configuration file
CONFIG_FILE="/lustre/scratch126/cellgen/team361/sb75/nichejepa/configs/model/${EXPERIMENT_NAME}_${GT_TYPE}.yaml"

# Set the output and error files
mkdir -p "${LOG_DIR}"
OUTPUT_FILE="${LOG_DIR}/${CORES}_%J.out"
ERROR_FILE="${LOG_DIR}/${CORES}_%J.err"

JOB_NAME="${EXPERIMENT_NAME}_${GT_TYPE}_${CORES}"

echo "Log directory: ${LOG_DIR}"
echo "Training ${GT_TYPE} model on ${EXPERIMENT_NAME} dataset using ${CORES} cores and ${N_GPU} GPUs..."
echo "Running script ${SCRIPT} with config file ${CONFIG_FILE}, devices ${DEVICES}, and run ID ${RUN_ID}..."

bsub \
    -G "${GROUP}" \
    -n "${CORES}" \
    -q "${QUEUE}" \
    -gpu "mode=exclusive_process:num=${N_GPU}" \
    -M "${RAM}" -R "select[mem>${RAM}] rusage[mem=${RAM}]" \
    -W "${TIME}" \
    -cwd "${CWD}" \
    -o "${OUTPUT_FILE}" \
    -e "${ERROR_FILE}" \
    -J "${JOB_NAME}" \
    "${SCRIPT}" "${CONFIG_FILE}" "${DEVICES}" "${RUN_ID}"
