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

This runs, in order: Q1 data prep, Q2 supervised learning, Q3 representation learning, Q4 foundation models.

### Run by stage

```bash
# Q1
sbatch data_prep/preprocess.sh
sbatch data_prep/exploratory.sh
sbatch data_prep/data_processing.sh
sbatch data_prep/horn_data_processing.sh

# Q2
sbatch supervised_learning/classic_ML_1.sh
sbatch supervised_learning/classic_ML_2.sh
sbatch supervised_learning/rnns.sh
sbatch supervised_learning/simple_transformer.sh
sbatch supervised_learning/horn_and_simple.sh

# Q3
sbatch rep_learning/rep_learn.sh

# Q4
sbatch foundation_models/predict_evaluate.sh
sbatch foundation_models/plotting.sh
sbatch foundation_models/chronos_pipeline.sh
```

## Expected Environment

- Repository path: `~/project1ml4h`
- Raw data path: `~/ml4h_data/p1`
- Cluster account: `ml4h`
- Python env used by scripts: `/cluster/courses/ml4h/jupyter/bin/activate`

**MAKE SURE ALL SCRIPTS GET DATA FROM THE SAME PLACE**


**FIX PYTHON ENVIRONMENT HANDLING SO ALL SCRIPTS RUN - CURRENTLY UMAPS AND ENTIRETY OF FINAL CHRONOS SCRIPT DONT RUN ON NORMAL ENV**

## Pipeline Overview (Q1-Q4)

### Q1: Data processing and exploration

- `data_prep/preprocessing.py`
	- Parses `{PatientID}.txt` into hourly grid (49 rows/patient).
	- Writes `data/processed/set_{a,b,c}.parquet`.
- `data_prep/exploratory.py`
	- Creates EDA plots and summary stats.
	- Writes under `figures/exploratory/`.
- `data_prep/data_processing.py`
	- Outlier cleaning, forward fill, leading-NaN defaults.
	- Log1p on skewed variables and robust scaling (median/IQR fit on set A).
	- Writes `data/processed_derived/set_*_{ffill,linear}.parquet`.
- `data_prep/horn_data_processing.py`
	- Tokenizes measurements into Horn-style tokens for Q2.3b.
	- Writes `data/horn_processed/set_*_horn.parquet` and variable maps.

### Q2: Supervised learning

- `supervised_learning/classic_ML_1.py`: last-timestep features
- `supervised_learning/classic_ML_2.py`: engineered aggregate features
- `supervised_learning/rnns.py`: uni/bi-LSTM ablations
- `supervised_learning/simple_transformer.py`: time-grid Transformer
- `supervised_learning/horn_and_simple.py`: time-grid vs Horn token Transformer

All summaries are saved in `supervised_learning/results/`.

### Q3: Representation learning

Single entrypoint: `rep_learning/rep_learn.sh`

Runs:
1. `rep_learning/pretrain_nce.py` (InfoNCE pretraining)
2. `rep_learning/linear_probe.py`
3. `rep_learning/label_scarce_predict.py`
4. `rep_learning/visualize_rep.py`

Outputs:
- Checkpoints: `checkpoints/encoder_best.pt`, `checkpoints/q3_2_label_scarce/*.pt`
- Embeddings: `checkpoints/embeddings_{train,val,test}.npz`
- Summaries: `rep_learning/results/q3_*/summary.json`
- Q3.2 table: `rep_learning/results/q3_2_label_scarce/comparison_table.csv`

### Q4: Foundation models

- `foundation_models/predict_evaluate.py`
	- LLM prompting (zero/few-shot flags) and LLM embedding linear probe.
	- Saves embeddings/labels as `.npy` in `foundation_models/`.
- `foundation_models/chronos_pipeline.py`
	- Chronos channel-wise embeddings + mean vs learned channel aggregation.
- `foundation_models/plotting.py`
	- t-SNE/UMAP plots from saved embedding arrays.

## Current Result Snapshot (Test Set C)

Numbers below are from saved result files/logs in this repo.

| Task | Model | Test AUROC | Test AUPRC | Source |
|---|---|---:|---:|---|
| Q2.1.1 | XGBoost (last timestep) | 0.8589 | 0.5382 | `supervised_learning/results/q2_1_1/summary.json` |
| Q2.1.2 | XGBoost (engineered features) | 0.8637 | 0.5542 | `supervised_learning/results/q2_1_2/summary.json` |
| Q2.2 | BiLSTM (best ablation: `bi_recency_6`) | 0.8259 | 0.4723 | `supervised_learning/results/q2_2_lstm_ablation/summary.json` |
| Q2.3a | Time-grid Transformer (selected best-by-val config) | 0.8207 | 0.4486 | `supervised_learning/results/q2_3a_transformer/summary.json` |
| Q2.3b | Horn-token Transformer (best config) | 0.8189 | 0.4818 | `supervised_learning/results/q2_3_compare_transformers/horn_tokens/summary.json` |
| Q3.1 | Linear probe on pretrained encoder | 0.8203 | 0.4490 | `rep_learning/results/q3_1_linear_probe/summary.json` |
| Q3.2 (full) | Best from-scratch model (Transformer) | 0.8293 | 0.4649 | `rep_learning/results/q3_2_label_scarce/comparison_table.csv` |
| Q4.2 | LLM embedding linear probe | 0.6447 | 0.2311 | `slurm_logs/foundation46046.out` |
