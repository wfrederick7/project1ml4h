#!/bin/bash
#SBATCH --job-name=p1_preprocess
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=process.out
#SBATCH --error=process.err

cd /home/$USER/project1ml4h/preprocessing

source /cluster/courses/ml4h/jupyter/bin/activate

python data_processing.py