# Fraud Scoring Baseline (Credit Card Fraud Detection)

Minimal, production-ish baseline for scoring fraud using the Kaggle **Credit Card Fraud Detection** dataset (`creditcard.csv` with columns `Time, V1..V28, Amount, Class`).  
It trains a model locally, saves artifacts, and scores a single transaction from JSON.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the dataset at `data/creditcard.csv`, then:

```bash
python src/train.py --data_path data/creditcard.csv
python src/predict.py --input_json examples/one_txn.json
```

## What gets produced

Artifacts are written to `./artifacts`:
- `model.pkl` — trained model
- `features.json` — list of feature columns used for training
- `metrics.json` — PR-AUC, ROC-AUC, precision/recall/F1, and metadata
- `run_config.json` — training config snapshot

## Imbalance handling

This baseline uses **class weighting** instead of downsampling:
- LightGBM/XGBoost: `scale_pos_weight = (#neg / #pos)` computed on the training split
- LogisticRegression fallback: `class_weight="balanced"`

This keeps the full dataset while weighting fraud examples more heavily.

## CLI details

Train:
```bash
python src/train.py --data_path data/creditcard.csv --test_size 0.2 --random_seed 42 --model_type auto --artifacts_dir artifacts
```

Predict:
```bash
python src/predict.py --artifacts_dir artifacts --input_json examples/one_txn.json --threshold 0.5
```

`--model_type` supports `auto`, `lightgbm`, `xgboost`, or `logreg` (auto prefers LightGBM).

## MLflow

Training logs to MLflow (params, metrics, artifacts) and registers the model as `fraud_scorer`.

Run UI:
```bash
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser.
