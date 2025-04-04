#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, ) gpu-lotfollahi
#BSUB -gpu 'mode=exclusive_process:num=1:block=yes' # request for exclusive access to gpu
#BSUB -G team361 # groupname for billing
#BSUB -o ../logs/%J.out # output file
#BSUB -e ../logs/%J.err # error file
#BSUB -R "span[ptile=6]"
#BSUB -M 256000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>256000] rusage[mem=256000]' # RAM memory part 1. Default: 100MB
#BSUB -n 13 # number of cores

# Load modules
. /usr/share/modules/init/bash
module load cuda-12.1.1
module load cellgen/conda

conda activate nichejepa_new
cd /lustre/scratch126/cellgen/team361/sb75/nichejepa

# Run main script
python main.py --fname /lustre/scratch126/cellgen/team361/sb75/nichejepa-reproducibility/config/model/human_cohort1_10m_cell_graph.yaml --devices cuda:0

echo "Finished script." 
