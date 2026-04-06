#!/bin/bash
set -euo pipefail

cd /home/$USER/project1ml4h

mkdir -p slurm_logs

echo "=== Data Preparation ==="
sbatch --wait data_prep/preprocess.sh
sbatch --wait data_prep/exploratory.sh
sbatch --wait data_prep/data_processing.sh
sbatch --wait data_prep/horn_data_processing.sh

echo "=== Supervised Learning ==="
sbatch --wait supervised_learning/classic_ML_1.sh
sbatch --wait supervised_learning/classic_ML_2.sh
sbatch --wait supervised_learning/rnns.sh
sbatch --wait supervised_learning/simple_transformer.sh
sbatch --wait supervised_learning/horn_and_simple.sh

echo "=== Representation Learning ==="
sbatch --wait rep_learning/rep_learn.sh

echo "=== Foundation Models ==="
sbatch --wait foundation_models/predict_evaluate.sh
sbatch --wait foundation_models/chronos_pipeline.sh
sbatch --wait foundation_models/plotting.sh

echo "All stages completed."
