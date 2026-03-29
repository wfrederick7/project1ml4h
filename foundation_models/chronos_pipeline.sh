#!/bin/bash
#SBATCH --job-name=chronos
#SBATCH --account=ml4h
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=chronos.out
#SBATCH --error=chronos.err

cd /home/wfrederick/project1ml4h/foundation_models

source /cluster/courses/ml4h/jupyter/bin/activate

export PYTHONPATH=/home/wfrederick/project1ml4h/libs:$PYTHONPATH
export HF_HOME=/home/wfrederick/project1ml4h/hf_cache

OLLAMA_MODELS=/cluster/courses/ml4h/llm/models OLLAMA_HOST=127.0.0.1:11435 /cluster/courses/ml4h/llm/bin/ollama serve &

sleep 10

python chronos_pipeline.py