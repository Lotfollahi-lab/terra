#!/bin/bash
#BSUB -q gpu-huge  # name of the partition to run job on (options: gpu-normal, gpu-huge, ) gpu-lotfollahi
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu
#BSUB -G team361 # groupname for billing
#BSUB -o /lustre/scratch126/cellgen/team361/mv10/jepa/wandb_log/nichejepa/J.out # output file
#BSUB -o /lustre/scratch126/cellgen/team361/mv10/jepa/wandb_log/nichejepa/J.err # error file
#BSUB -R "span[ptile=6]"
#BSUB -M 30000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>30000] rusage[mem=30000]' # RAM memory part 1. Default: 100MB
#BSUB -n 6 # number of cores

# Load modules
. /usr/share/modules/init/bash
module load cuda-12.1.1
source /nfs/team361/mv10/.jepa/bin/activate

cd /lustre/scratch126/cellgen/team361/mv10/jepa/wandb_log/nichejepa

python main.py --fname configs/merfish_300k.yaml --devices cuda:0 cuda:1

