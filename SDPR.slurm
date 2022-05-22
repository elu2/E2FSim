#!/bin/bash

#SBATCH --job-name=SDPR
#SBATCH --mincpus=94
#SBATCH --ntasks=1
#SBATCH --mem=5GB
#SBATCH --time=4:00:00
#SBATCH --partition=standard
#SBATCH --account=
#SBATCH --array 0-99

python3 SDPR.py ${SLURM_ARRAY_TASK_ID}
