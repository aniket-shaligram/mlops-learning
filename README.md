# Fraud Scoring Baseline (Synthetic)

Minimal, production-ish baseline for scoring fraud using an interpretable synthetic dataset.
It trains a model locally, saves artifacts, and scores a single transaction from JSON.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Synthetic dataset generator

Generate an interpretable fraud dataset (1M rows by default) with velocity features,
new device/IP signals, geo mismatch, merchant risk, label delay, and drift.

```bash
python src/synth/generate_synth.py --rows 1000000 --format parquet
```

Parquet output requires `pyarrow` (included in `requirements.txt`).

Smoke test (10k rows):
```bash
python src/synth/generate_synth.py --smoke_test --format csv
```

This writes:
- `data/synth_transactions.(csv|parquet)`
- `examples/one_txn.json` (sample input for prediction)
- `data/synth_profiles/` (user/merchant/device/ip profiles)

Train/predict on synthetic data:
```bash
python src/train.py --data_path data/synth_transactions.parquet --dataset_type synth --artifacts_dir artifacts_synth
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json
```
`--dataset_type synth` enables one-hot encoding for categorical fields (currency, country, channel).

## What gets produced

Artifacts are written to `./artifacts_synth`:
- `model.pkl` — trained model
- `features.json` — list of feature columns used for training
- `metrics.json` — PR-AUC, ROC-AUC, precision/recall/F1, optimal threshold, and metadata
- `run_config.json` — training config snapshot

## Imbalance handling

This baseline uses **class weighting** instead of downsampling:
- LightGBM/XGBoost: `scale_pos_weight = (#neg / #pos)` computed on the training split
- LogisticRegression fallback: `class_weight="balanced"`

This keeps the full dataset while weighting fraud examples more heavily.

## CLI details

Train:
```bash
python src/train.py --data_path data/synth_transactions.parquet --dataset_type synth --test_size 0.2 --random_seed 42 --model_type auto --artifacts_dir artifacts_synth
```

Predict:
```bash
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json
```

If you want to override the decision threshold:
```bash
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --threshold 0.5
```

`--model_type` supports `auto`, `lightgbm`, `xgboost`, or `logreg` (auto prefers LightGBM).

## MLflow

Training logs to MLflow (params, metrics, artifacts) and logs the model under the run in `./mlruns`.
You can add model registry later if needed.

Run UI:
```bash
mlflow ui
```

Open `http://127.0.0.1:5000` in your browser.
