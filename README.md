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

Parquet output is the default format and requires `pyarrow` (included in `requirements.txt`).

Smoke test (10k rows):
```bash
python src/synth/generate_synth.py --smoke_test --format csv
```

This writes:
- `data/synth_transactions.(csv|parquet)`
- `examples/one_txn.json` (sample input for prediction, includes entity IDs)
- `data/synth_profiles/` (user/merchant/device/ip profiles)

Train/predict on synthetic data (payload-only, no Feast):
```bash
python src/train.py --data_path data/synth_transactions.parquet --dataset_type synth --artifacts_dir artifacts_synth
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --use_feast false
```
`--dataset_type synth` enables one-hot encoding for categorical fields (currency, country, channel).

## Feast online features (Phase 1)

Phase 1 keeps training unchanged and uses Feast (Redis online store) at inference time.
For now, **transaction context** fields stay in the request payload:
`amount`, `hour_of_day`, `geo_mismatch`, `is_new_device`, `is_new_ip`, plus the categorical fields
`currency`, `country`, `channel`, and the remaining model features such as
`distance_from_home_km` and `drift_phase`.

Start Redis:
```bash
docker run -p 6379:6379 redis
```
Or with Homebrew:
```bash
brew install redis
brew services start redis
```

Generate synthetic dataset in Parquet:
```bash
python src/synth/generate_synth.py --rows 1000000 --format parquet
```

Apply Feast definitions:
```bash
cd feast_repo && feast apply
```
The registry is stored at `feast_repo/data/feast_registry.db` (ignored by git).

Materialize to the online store (use a current UTC timestamp):
```bash
cd feast_repo
feast materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"
```
Optional: materialize up to the latest event in the Parquet file:
```bash
python - <<'PY'
import pandas as pd

ts = pd.read_parquet("data/synth_transactions.parquet")["event_ts"].max()
print(pd.to_datetime(ts, utc=True).strftime("%Y-%m-%dT%H:%M:%S"))
PY
```

Run prediction with Feast:
```bash
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --dataset_type synth --use_feast true
```
Ids-only request mode (entity ids + minimal context in payload):
```bash
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --dataset_type synth --use_feast true --ids_only_request true
```
If you see a missing non-Feast context error, include those fields in the payload
(e.g. `is_new_device`, `is_new_ip`, `geo_mismatch`, `distance_from_home_km`, `drift_phase`).

Minimal validation:
```bash
cd feast_repo && feast entities list
cd feast_repo && feast feature-views list
```

Quick online fetch check:
```bash
python - <<'PY'
from feast import FeatureStore

store = FeatureStore(repo_path="feast_repo")
features = store.get_online_features(
    features=[
        "user_features:user_txn_count_5m",
        "merchant_features:merchant_chargeback_rate_30d",
        "device_features:device_risk_score",
        "ip_features:ip_risk_score",
    ],
    entity_rows=[{"user_id": 1, "merchant_id": 1, "device_id": 1, "ip_id": 1}],
).to_dict()
print(features)
PY
```

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
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --use_feast false
```
To use Feast at inference time:
```bash
python src/predict.py --artifacts_dir artifacts_synth --input_json examples/one_txn.json --dataset_type synth --use_feast true
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
