#!/bin/bash
#SBATCH --job-name=classic_ml_1
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/classic_ml_1.out
#SBATCH --error=slurm_logs/classic_ml_1.err

cd /home/$USER/project1ml4h

source /cluster/courses/ml4h/jupyter/bin/activate

python supervised_learning/classic_ML_1.py