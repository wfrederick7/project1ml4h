# ICU Time-Series Mortality Prediction

## How to Run

### Full pipeline

```bash
sbatch run_all.sh
```

This runs all stages sequentially (Q1–Q4) via `sbatch --wait`. On first run it creates a Python venv at `/work/scratch/$USER/.venv` and installs `requirements.txt`.

### Run by stage

```bash
# Q1 - Data Preparation
sbatch data_prep/preprocess.sh
sbatch data_prep/exploratory.sh
sbatch data_prep/data_processing.sh
sbatch data_prep/horn_data_processing.sh

# Q2 - Supervised Learning
sbatch supervised_learning/classic_ML_1.sh
sbatch supervised_learning/classic_ML_2.sh
sbatch supervised_learning/rnns.sh
sbatch supervised_learning/simple_transformer.sh
sbatch supervised_learning/horn_and_simple.sh

# Q3 - Representation Learning
sbatch rep_learning/rep_learn.sh

# Q4 - Foundation Models
sbatch foundation_models/predict_evaluate_zero.sh
sbatch foundation_models/predict_evaluate_few.sh
sbatch foundation_models/predict_evaluate_emb.sh
sbatch foundation_models/chronos_pipeline.sh
sbatch foundation_models/plotting.sh
```

Stages must run in order (Q1 before Q2, etc.). Within each stage, scripts should also run sequentially.

## Environment

- **Cluster account:** `ml4h`
- **Repository path:** `~/project1ml4h`
- **Raw data path:** `~/ml4h_data/p1`
- **Python venv:** `/work/scratch/$USER/.venv` — created automatically by `run_all.sh`, activated in each script via `source activate_env.sh`
- **Q4 (LLM scripts):** Start a local Ollama server (`127.0.0.1:11435`) for LLaMA 3.1. The Chronos script additionally sets `HF_HOME` and `PYTHONPATH` for the local `libs/` directory.
