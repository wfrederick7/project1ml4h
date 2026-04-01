#!/bin/bash
#SBATCH --job-name=p1_pretrain
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=pretrain.out
#SBATCH --error=pretrain.err

cd /home/$USER/project1ml4h

source /cluster/courses/ml4h/jupyter/bin/activate

python supervised_learning/rnns.py