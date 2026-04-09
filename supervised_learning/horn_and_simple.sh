#!/bin/bash
#SBATCH --job-name=horn_and_simple
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/horn_and_simple.out
#SBATCH --error=slurm_logs/horn_and_simple.err

cd /home/$USER/project1ml4h

source "$SLURM_SUBMIT_DIR/activate_env.sh"

python supervised_learning/horn_and_simple.py