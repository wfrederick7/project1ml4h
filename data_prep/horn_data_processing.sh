#!/bin/bash
#SBATCH --job-name=horn_data
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/horn_data.out
#SBATCH --error=slurm_logs/horn_data.err

cd /home/$USER/project1ml4h/data_prep

source /cluster/courses/ml4h/jupyter/bin/activate

python horn_data_processing.py