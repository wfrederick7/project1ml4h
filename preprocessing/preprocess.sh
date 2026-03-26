#!/bin/bash
#SBATCH --job-name=p1_preprocess
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --exclude=studgpu-node*
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err

source /cluster/courses/ml4h/project1env/bin/activate

python data/preprocessing.py