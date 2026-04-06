#!/bin/bash
#SBATCH --job-name=foundation_plotting
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/plotting.out
#SBATCH --error=slurm_logs/plotting.err

set -euo pipefail

cd /home/$USER/project1ml4h/foundation_models

source /cluster/courses/ml4h/jupyter/bin/activate

python plotting.py
