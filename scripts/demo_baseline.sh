#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_BASELINE="${DEMO_SECONDS_BASELINE:-30}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_SCORE_URL="${DEMO_SCORE_URL:-http://localhost:8083/score}"
DEMO_UI_URL="${DEMO_UI_URL:-http://localhost:8083/ui}"
DEMO_FEAST_REPO="${DEMO_FEAST_REPO:-feast_repo}"

echo "==> Starting infra"
docker compose -f ops/docker-compose.yml up -d postgres redis rabbitmq serving-v1

echo "==> Waiting for serving-v1"
python src/demo/wait_for_http.py --url "http://localhost:8083/health" --timeout 120

echo "==> Start GX worker"
mkdir -p logs
python src/gx/validate_and_forward.py > logs/gx_worker.log 2>&1 &
sleep 3

echo "==> Generate baseline synthetic data"
python src/synth/generate_synth.py --rows 200000 --format parquet --drift_start_day 9999

export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5432}"
export POSTGRES_DB="${POSTGRES_DB:-fraud_poc}"
export POSTGRES_USER="${POSTGRES_USER:-fraud}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-fraud}"
export FEAST_OFFLINE_SOURCE="${FEAST_OFFLINE_SOURCE:-postgres}"

echo "==> Publish baseline events"
python src/eventing/publish_transactions.py \
  --data_path data/synth_transactions.parquet \
  --rate 3000 \
  --max_events 20000

if [ -d "${DEMO_FEAST_REPO}" ]; then
  echo "==> Feast apply"
  (cd "${DEMO_FEAST_REPO}" && feast apply) || true
fi

echo "==> Materialize Feast (best effort)"
python src/feast_materialize.py --start now-1d --end now || true

echo "==> Train v1"
python src/train_phase4.py --start now-30d --end now --registered_model_name fraud-risk
mkdir -p model_bundles/v1
cp -R artifacts_phase4/* model_bundles/v1/ || true
docker compose -f ops/docker-compose.yml up -d --build serving-v1
python src/demo/wait_for_http.py --url "http://localhost:8083/health" --timeout 120

echo "==> Baseline traffic"
python src/demo/publish_and_score.py --url "${DEMO_SCORE_URL}" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_BASELINE}" --mode baseline --seed "${DEMO_SEED}"

echo "==> Monitoring snapshot"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Open these URLs"
echo "Serving UI: ${DEMO_UI_URL}"
echo "Reports: ${DEMO_UI_URL%/ui}/reports/index.html"
