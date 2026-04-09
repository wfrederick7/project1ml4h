#!/bin/bash
#SBATCH --job-name=foundation
#SBATCH --account=ml4h
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm_logs/foundation_embedding.out
#SBATCH --error=slurm_logs/foundation_embedding.err

# Activate the course environment
module load cuda/12.6
cd /home/$USER/project1ml4h/foundation_models
source /cluster/courses/ml4h/jupyter/bin/activate

# Start ollama on a custom port
OLLAMA_MODELS=/cluster/courses/ml4h/llm/models OLLAMA_HOST=127.0.0.1:11435 /cluster/courses/ml4h/llm/bin/ollama serve &

sleep 10


# Check what models are available on YOUR server
OLLAMA_HOST=127.0.0.1:11435 /cluster/courses/ml4h/llm/bin/ollama list

python predict_evaluate.py --run-embeddings