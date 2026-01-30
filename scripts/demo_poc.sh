#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_BASELINE="${DEMO_SECONDS_BASELINE:-30}"
DEMO_SECONDS_DRIFT="${DEMO_SECONDS_DRIFT:-30}"
DEMO_SECONDS_CANARY="${DEMO_SECONDS_CANARY:-120}"
DEMO_DELAY_MINUTES="${DEMO_DELAY_MINUTES:-0}"
DEMO_BATCH_SIZE="${DEMO_BATCH_SIZE:-500}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_ROUTER_URL="${DEMO_ROUTER_URL:-http://localhost:8080}"
DEMO_FEAST_REPO="${DEMO_FEAST_REPO:-feast_repo}"

echo "This script has been split. Use:"
echo "  ./scripts/demo_baseline.sh"
echo "  ./scripts/demo_drift.sh"
echo "  ./scripts/demo_retrain.sh"
echo "  ./scripts/demo_canary.sh (separate canary story)"
