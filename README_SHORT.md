# Fraud PoC — Quick End-to-End

Minimal steps to run the PoC locally and hit the scoring endpoint.

## 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Start infra
```bash
docker compose -f ops/docker-compose.yml up -d postgres redis rabbitmq
```

## Demo (single command)
```bash
chmod +x scripts/demo_poc.sh scripts/feast_story/*.sh
./scripts/demo_poc.sh
```

## 3) Run GX worker (terminal 1)
```bash
python src/gx/validate_and_forward.py
```

## 4) Generate + publish events (terminal 2)
```bash
python src/synth/generate_synth.py --rows 100000 --format parquet
python src/eventing/publish_transactions.py --data_path data/synth_transactions.parquet --rate 5000 --max_events 0
```

## 5) Feast apply + materialize
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=fraud_poc
export POSTGRES_USER=fraud
export POSTGRES_PASSWORD=fraud
export FEAST_OFFLINE_SOURCE=postgres

cd feast_repo && feast apply
cd ..
python src/feast_materialize.py --start now-7d --end now
```

## 6) Train Phase 4 + start serving
```bash
python src/train_phase4.py --start now-30d --end now --registered_model_name fraud-risk
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload
```
Open `http://localhost:8080/ui`.

## 7) Test scoring
```bash
curl -X POST http://localhost:8080/score \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"evt_1","event_type":"txn.created","event_ts":"2026-01-29T00:00:00Z","user_id":1,"merchant_id":10,"device_id":5,"ip_id":7,"amount":120.5,"currency":"USD","country":"US","channel":"web","drift_phase":0}'
```

Metrics: `http://localhost:8080/metrics`

Demo UI: `http://localhost:8080/ui`

Generate 200 scoring events (for monitoring reports):
```bash
python src/monitoring/seed_decisions.py --count 200
```

## Canary / Shadow router (optional, off by default)
```bash
docker compose -f ops/docker-compose.yml up --build router serving-v1 serving-v2
```

## Monitoring (optional)
```bash
docker compose -f ops/docker-compose.yml up -d prometheus grafana
python src/monitoring/run_all.py --ref_hours 24 --cur_hours 1
```
Prometheus targets: `http://localhost:9090/targets` (expect `host.docker.internal:8080`).

Open latest report (macOS):
```bash
open "$(ls -t monitoring/reports/evidently/*/input_drift.html | head -1)"
```
