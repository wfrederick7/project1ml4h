#!/bin/bash
#SBATCH --job-name=p1_exploratory
#SBATCH --account=ml4h
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=exploratory.out
#SBATCH --error=exploratory.err

set -euo pipefail

cd /home/$USER/project1ml4h/data_prep

source /cluster/courses/ml4h/jupyter/bin/activate

python exploratory.py
