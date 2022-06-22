#!/bin/bash

#SBATCH --job-name=SDPR
#SBATCH --mincpus=94
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=4:00:00
#SBATCH --partition=standard
#SBATCH --account=
#SBATCH --array=0-99

cd /xdisk/guangyao/elu2/E2FSim/

module load python/3.9/3.9.10

python3 SDPR.py ${SLURM_ARRAY_TASK_ID}
