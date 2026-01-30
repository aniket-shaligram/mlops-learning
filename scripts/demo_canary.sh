#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_CANARY="${DEMO_SECONDS_CANARY:-120}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_ROUTER_URL="${DEMO_ROUTER_URL:-http://localhost:8080}"

echo "==> Start router + serving v1/v2"
docker compose -f ops/docker-compose.yml up -d --build router serving-v1 serving-v2
python src/demo/wait_for_http.py --url "${DEMO_ROUTER_URL}/api/stats" --timeout 120

echo "==> Enable canary"
python src/demo/try_enable_canary.py --base_url "${DEMO_ROUTER_URL}" --canary_percent 20 || true

echo "==> Send traffic via router"
python src/demo/publish_and_score.py --url "${DEMO_ROUTER_URL}/score" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_CANARY}" --mode baseline --seed "${DEMO_SEED}"

echo "==> Compare models"
python src/demo/compare_models.py --minutes 30

echo "==> Open these URLs"
echo "Router UI: ${DEMO_ROUTER_URL}/ui"
echo "Shadow comparisons: ${DEMO_ROUTER_URL}/api/shadow-comparisons"
