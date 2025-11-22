#!/bin/bash
CONFIG_FILE=$1 # path to the config file
DEVICES=$2 # GPU devices
RUN_ID=$3

. /usr/share/modules/init/bash
module load cuda-12.1.1
module load cellgen/conda
conda activate nichejepa_new

python /lustre/scratch126/cellgen/team361/sb75/nichejepa/src/app/main.py \
    --fname ${CONFIG_FILE} \
    --devices ${DEVICES} \
    --run_id ${RUN_ID}

echo "Finished script." 