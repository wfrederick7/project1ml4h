#!/bin/bash
#SBATCH --job-name=p1_preprocess
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err

cd /home/$USER/project1ml4h/data_prep

source /cluster/courses/ml4h/jupyter/bin/activate

python preprocessing.py