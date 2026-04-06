#!/bin/bash
#SBATCH --job-name=rnns
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/rnns.out
#SBATCH --error=slurm_logs/rnns.err

cd /home/$USER/project1ml4h

source /cluster/courses/ml4h/jupyter/bin/activate

python supervised_learning/rnns.py