#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/preprocess.out
#SBATCH --error=slurm_logs/preprocess.err

cd /home/$USER/project1ml4h/data_prep

source "$SLURM_SUBMIT_DIR/activate_env.sh"

python preprocessing.py