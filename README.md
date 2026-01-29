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
- `examples/one_txn_ids_only.json` (ids-only request example)
- `data/synth_profiles/` (user/merchant/device/ip profiles)

## Feast features

Training uses Feast offline historical features for the synthetic dataset. Inference
fetches Feast features online. In ids-only mode, the request must include entity ids
(`user_id`, `merchant_id`, `device_id`, `ip_id`) plus `amount`, `country`, `channel`,
and `event_ts`. Remaining model fields are defaulted to: `currency="USD"`,
`distance_from_home_km=0.0`, `geo_mismatch=0`, `is_new_device=0`, `is_new_ip=0`,
`drift_phase=0`. Full-payload mode expects all model features in the request.

## Eventing + GX (RabbitMQ → GX → Postgres)

This pipeline validates transactions with Great Expectations and writes to
`txn_validated` or `txn_quarantine`.

### 1) Start infra
```bash
docker compose -f ops/docker-compose.yml up -d
```

RabbitMQ UI: `http://localhost:15672` (guest/guest).
If Redis is already running on your machine, stop it or remove the container using port `6379`.

### 2) Initialize schema
Schema is auto-applied on first start via `/docker-entrypoint-initdb.d`.
If you need to re-init the database (drops data), reset the volume:
```bash
docker compose -f ops/docker-compose.yml down -v
docker compose -f ops/docker-compose.yml up -d
```

To apply manually inside the container:
```bash
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -f /docker-entrypoint-initdb.d/002_create_validated_quarantine.sql
```

### 3) Run the GX worker
```bash
python src/gx/validate_and_forward.py
```

### 4) Generate synthetic data
```bash
python src/synth/generate_synth.py --rows 100000 --format parquet
```

### 5) Publish events
```bash
python src/eventing/publish_transactions.py \
  --data_path data/synth_transactions.parquet \
  --rate 5000 \
  --max_events 0
```

### 6) Verify results
```bash
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c "select count(*) from txn_validated;"
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c "select count(*) from txn_quarantine;"
```

### Notes
- Messages are durable and published to exchange `tx.events` with routing key `txn.created`.
- Invalid messages are nacked and routed to `tx.raw.dlq`.

### How to test quickly
```bash
python src/synth/generate_synth.py --rows 10000 --format parquet
python src/eventing/publish_transactions.py --data_path data/synth_transactions.parquet --rate 2000 --max_events 10000
```

RabbitMQ queues:
- `tx.validated.q` (routing key `txn.validated`)
- `tx.quarantine.q` (routing key `txn.quarantined`)

## Offline training with Feast

Offline training uses Feast historical features for the synthetic dataset and does **not**
require Redis.

Generate a Parquet dataset:
```bash
python src/synth/generate_synth.py --rows 10000 --format parquet
```

Apply Feast definitions:
```bash
cd feast_repo && feast apply
```

Train with Feast offline features:
```bash
python src/train.py \
  --dataset_type synth \
  --data_path data/synth_transactions.parquet \
  --use_feast_offline true \
  --artifacts_dir artifacts_synth_feast
```

## Feast offline training (recommended dev size)

Use the 10k smoke test for quick checks, but standardize dev runs on a 100k slice.
Generate a full Parquet dataset (rows can be 1M; training uses a 100k slice):
```bash
python src/synth/generate_synth.py --rows 1000000 --format parquet
```

Apply Feast definitions:
```bash
cd feast_repo && feast apply
```

Train using the 100k slice:
```bash
python src/train.py \
  --dataset_type synth \
  --data_path data/synth_transactions.parquet \
  --use_feast_offline true \
  --dev_100k true \
  --artifacts_dir artifacts_synth_feast_100k
```

## What gets produced

Artifacts are written to `./artifacts_synth_feast`:
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
python src/train.py --data_path data/synth_transactions.parquet --dataset_type synth --use_feast_offline true --test_size 0.2 --random_seed 42 --model_type auto --artifacts_dir artifacts_synth_feast
```

Predict (ids-only request):
```bash
python src/predict.py --artifacts_dir artifacts_synth_feast --input_json examples/one_txn_ids_only.json --dataset_type synth --use_feast true --ids_only_request true
```

If you want to override the decision threshold:
```bash
python src/predict.py --artifacts_dir artifacts_synth_feast --input_json examples/one_txn_ids_only.json --dataset_type synth --use_feast true --ids_only_request true --threshold 0.5
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
