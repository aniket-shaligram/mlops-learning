# Fraud Scoring Baseline (Synthetic)

Minimal, production-ish baseline for scoring fraud using an interpretable synthetic dataset.
It trains a model locally, saves artifacts, and scores a single transaction from JSON.

For documentation:
- `README_DETAILED.md` — full, step-by-step guide
- `README_SHORT.md` — minimal end-to-end PoC steps

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Optional: generate synthetic data

```bash
python src/synth/generate_synth.py --rows 100000 --format parquet
```

Writes `data/synth_transactions.(csv|parquet)` and `examples/one_txn_ids_only.json`.
Parquet output requires `pyarrow` (included in `requirements.txt`).

## End-to-end runbook

This flow validates transactions with Great Expectations, stores them in Postgres,
materializes features into Redis, and trains a model with Feast offline features.

### 1) Start infra
```bash
docker compose -f ops/docker-compose.yml up -d
```

RabbitMQ UI: `http://localhost:15672` (guest/guest).
If Redis is already running on your machine, stop it or remove the container using port `6379`.

### 2) Initialize schema
Schema is auto-applied on first start via `/docker-entrypoint-initdb.d` (002 script on a fresh volume).
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

### 6) Verify Postgres writes
```bash
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c "select count(*) from txn_validated;"
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c "select count(*) from txn_quarantine;"
```

### 7) Configure Feast env vars
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=fraud_poc
export POSTGRES_USER=fraud
export POSTGRES_PASSWORD=fraud
export FEAST_OFFLINE_SOURCE=postgres
```

If you want Feast to read features from the local Parquet file instead of Postgres,
set `FEAST_OFFLINE_SOURCE=file` and re-run `feast apply`.

### 8) Apply Feast definitions
```bash
cd feast_repo && feast apply
```

### 9) Materialize Postgres → Redis
```bash
python src/feast_materialize.py --start now-1d --end now
```

### 10) Verify online feature fetch
```bash
python - <<'PY'
import os
import sys

repo_root = "/Users/aniket/Talentica/Learning"
sys.path.append(repo_root)

import psycopg2
from feast import FeatureStore
from feast_client import FEAST_FEATURE_REFS

conn = psycopg2.connect("postgresql://fraud:fraud@localhost:5432/fraud_poc")
cur = conn.cursor()
cur.execute(
    "select user_id, merchant_id, device_id, ip_id from txn_validated order by event_ts desc limit 1"
)
row = cur.fetchone()
if not row:
    raise SystemExit("txn_validated is empty; publish events first.")

entity_row = {
    "user_id": int(row[0]),
    "merchant_id": int(row[1]),
    "device_id": int(row[2]),
    "ip_id": int(row[3]),
}

store = FeatureStore(repo_path="feast_repo")
print(store.get_online_features(FEAST_FEATURE_REFS, [entity_row]).to_dict())
conn.close()
PY
```

### 11) Train with Feast offline features
```bash
python src/train.py \
  --dataset_type synth \
  --data_path data/synth_transactions.parquet \
  --use_feast_offline true \
  --offline_source validated_db \
  --artifacts_dir artifacts_synth_feast
```

### 12) Train (recommended dev size)
Use the 10k smoke test for quick checks, but standardize dev runs on a 100k slice.
```bash
python src/synth/generate_synth.py --rows 1000000 --format parquet
python src/train.py \
  --dataset_type synth \
  --data_path data/synth_transactions.parquet \
  --use_feast_offline true \
  --offline_source validated_db \
  --dev_100k true \
  --artifacts_dir artifacts_synth_feast_100k
```

### Notes
- Messages are durable and published to exchange `tx.events` with routing key `txn.created`.
- Invalid messages are nacked and routed to `tx.raw.dlq`.
- RabbitMQ queues: `tx.validated.q` (`txn.validated`), `tx.quarantine.q` (`txn.quarantined`).

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

## Phase 4 training (multi-model)

Run the multi-model trainer (rules + GBDT + anomaly) and promote the champion:
```bash
python src/train_phase4.py --start now-30d --end now --registered_model_name fraud-risk
```

Champion storage:
- MLflow registry if configured (stage set via `--promote_stage`)
- Local fallback under `registry/fraud-risk/champion/`

## Serving (Phase 5)

Prereqs:
- Redis running
- Feast applied + materialized
- Phase 4 trained (`artifacts_phase4/champion.json` exists)

Run:
```bash
python src/train_phase4.py --start now-30d --end now --registered_model_name fraud-risk
python src/feast_materialize.py --start now-7d --end now
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload
```

Test:
```bash
curl -X POST http://localhost:8080/score \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"evt_1","event_type":"txn.created","event_ts":"2026-01-29T00:00:00Z","user_id":1,"merchant_id":10,"device_id":5,"ip_id":7,"amount":120.5,"currency":"USD","country":"US","channel":"web","drift_phase":0}'
```

Metrics: `http://localhost:8080/metrics`

### Demo UI
```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload
```
Open `http://localhost:8080/ui` and use the form to score transactions.
