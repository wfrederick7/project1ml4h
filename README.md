# ICU Time-Series Mortality Prediction (PhysioNet 2012)

End-to-end ML4H project pipeline for predicting in-hospital ICU mortality from the first 48h of stay.

- Task: binary classification (`label` = in-hospital death)
- Metrics: AUROC (`roc_auc_score`) and AUPRC (`average_precision_score`)
- Splits: Set A (train), Set B (val), Set C (test)
- Inputs: 37 dynamic + 4 static variables (ICUType excluded)

## Quick Start

### Full pipeline (cluster)

```bash
sbatch run_all.sh
```

This runs all stages sequentially via `sbatch --wait`: Q1 data prep, Q2 supervised learning, Q3 representation learning, Q4 foundation models. On first run it creates a venv at `/work/scratch/$USER/.venv` and installs `requirements.txt`.

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
sbatch foundation_models/plotting.sh
sbatch foundation_models/chronos_pipeline.sh
```

## Expected Environment

- Repository path: `~/project1ml4h`
- Raw data path: `~/ml4h_data/p1`
- Cluster account: `ml4h`
- Python venv: `/work/scratch/$USER/.venv` (created automatically by `run_all.sh`, activated via `activate_env.sh`)
- Foundation model scripts (Q4 zero-shot, embedding) also source `/cluster/courses/ml4h/jupyter/bin/activate` and start a local Ollama server for LLaMA 3.1

## Pipeline Overview (Q1-Q4)

### Q1: Data Processing and Exploration

- `data_prep/preprocessing.py`
	- Parses `{PatientID}.txt` into hourly grid (49 rows/patient).
	- Writes `data/processed/set_{a,b,c}.parquet`.
- `data_prep/exploratory.py`
	- Creates EDA plots and summary stats.
	- Writes under `figures/exploratory/`.
- `data_prep/data_processing.py`
	- Outlier cleaning, forward fill, leading-NaN defaults from set A.
	- Log1p on skewed variables and robust scaling (median/IQR fit on set A).
	- Writes `data/processed_derived/set_*_{ffill,linear}.parquet`.
- `data_prep/horn_data_processing.py`
	- Tokenizes measurements into Horn-style tokens for Q2.3b.
	- Writes `data/horn_processed/set_*_horn.parquet` and variable maps.

### Q2: Supervised Learning

- `supervised_learning/classic_ML_1.py` (Q2.1.1): Logistic Regression, Random Forest, XGBoost on last-timestep features.
- `supervised_learning/classic_ML_2.py` (Q2.1.2): Same models on engineered aggregate features (min, max, mean, std, slope, delta, etc.).
- `supervised_learning/rnns.py` (Q2.2): Uni/bidirectional LSTM with multiple output aggregation strategies (last, mean, max, recency-weighted).
- `supervised_learning/simple_transformer.py` (Q2.3a): Time-grid Transformer with hyperparameter sweep.
- `supervised_learning/horn_and_simple.py` (Q2.3b): Grid vs Horn-token Transformer comparison.

All summaries are saved in `supervised_learning/results/`.

### Q3: Representation Learning

Single entrypoint: `rep_learning/rep_learn.sh`

Runs:
1. `rep_learning/pretrain_nce.py` — InfoNCE contrastive pretraining of a BiLSTM encoder (augmentations: time-crop, Gaussian noise, channel dropout).
2. `rep_learning/linear_probe.py` — Frozen-encoder logistic probe.
3. `rep_learning/label_scarce_predict.py` — Label scarcity experiments (N=100, 500, 1000, full) comparing from-scratch models vs pretrained probe.
4. `rep_learning/visualize_rep.py` — t-SNE/UMAP visualisations + clustering metrics (silhouette, DB, ARI, NMI).
5. `rep_learning/plot_appendix_figures.py` — Appendix figures (pretraining curves, label scarcity plots).

Outputs:
- Checkpoints: `checkpoints/encoder_best.pt`, `checkpoints/q3_2_label_scarce/*.pt`
- Embeddings: `checkpoints/embeddings_{train,val,test}.npz`
- Summaries: `rep_learning/results/q3_*/summary.json`
- Q3.2 table: `rep_learning/results/q3_2_label_scarce/comparison_table.csv`

### Q4: Foundation Models

- `foundation_models/predict_evaluate.py`
	- LLM prompting via local Ollama (LLaMA 3.1). Three modes selected by flags:
		- `--run-zero-shot`: zero-shot binary/ordinal mortality prediction.
		- `--run-few-shot`: few-shot (6 examples) binary/ordinal prediction.
		- `--run-embeddings`: LLM embedding extraction + linear probe.
	- Each mode has its own shell script (`predict_evaluate_zero.sh`, `predict_evaluate_few.sh`, `predict_evaluate_emb.sh`).
- `foundation_models/chronos_pipeline.py`
	- Amazon Chronos time-series foundation model: channel-wise embeddings with mean vs learned aggregation.
- `foundation_models/plotting.py`
	- t-SNE/UMAP plots from saved LLM embedding arrays.
