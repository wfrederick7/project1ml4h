#!/bin/bash
#SBATCH --job-name=rep_learning
#SBATCH --account=ml4h
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/rep_learning.out
#SBATCH --error=slurm_logs/rep_learning.err

set -euo pipefail

cd /home/$USER/project1ml4h

source /cluster/courses/ml4h/jupyter/bin/activate

python rep_learning/pretrain_nce.py

python rep_learning/linear_probe.py

python rep_learning/label_scarce_predict.py

python rep_learning/visualize_rep.py

python rep_learning/plot_appendix_figures.py