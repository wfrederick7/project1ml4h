#!/bin/bash
#SBATCH --job-name=simple_transformer
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/simple_transformer.out
#SBATCH --error=slurm_logs/simple_transformer.err

cd /home/$USER/project1ml4h

source /cluster/courses/ml4h/jupyter/bin/activate

python supervised_learning/simple_transformer.py