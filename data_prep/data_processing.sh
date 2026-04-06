#!/bin/bash
#SBATCH --job-name=process
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/process.out
#SBATCH --error=slurm_logs/process.err

cd /home/$USER/project1ml4h/data_prep

source /cluster/courses/ml4h/jupyter/bin/activate

python data_processing.py