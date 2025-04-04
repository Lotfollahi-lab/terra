#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, ) gpu-lotfollahi
#BSUB -gpu 'mode=exclusive_process:num=1:block=yes' # request for exclusive access to gpu
#BSUB -G team361 # groupname for billing
#BSUB -o ../logs/run_inference/%J.out # output file
#BSUB -e ../logs/run_inference/%J.err # error file
#BSUB -R "span[ptile=6]"
#BSUB -M 128000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>128000] rusage[mem=128000]' # RAM memory part 1. Default: 100MB
#BSUB -n 6 # number of cores

# Load modules
. /usr/share/modules/init/bash
module load cuda-12.1.1
module load cellgen/conda

conda activate nichejepa_new
cd /lustre/scratch126/cellgen/team361/sb75/nichejepa

python run_inference.py --fname configs/merfish_2_gene_panels.yaml

echo "Finished script." 
