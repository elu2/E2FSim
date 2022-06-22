#!/bin/bash

#SBATCH --job-name=SeedSim
#SBATCH --mincpus=94
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=16:00:00
#SBATCH --partition=standard
#SBATCH --account=

cd /xdisk/guangyao/elu2/E2FSim/

module load python/3.9/3.9.10

python3 SeedSim.py
