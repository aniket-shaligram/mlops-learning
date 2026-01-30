#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_DRIFT="${DEMO_SECONDS_DRIFT:-30}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_SCORE_URL="${DEMO_SCORE_URL:-http://localhost:8083/score}"
DEMO_UI_URL="${DEMO_UI_URL:-http://localhost:8083/ui}"
DEMO_DELAY_MINUTES="${DEMO_DELAY_MINUTES:-0}"
DEMO_BATCH_SIZE="${DEMO_BATCH_SIZE:-500}"

echo "==> Drift dataset"
python src/synth/generate_synth.py --rows 200000 --format parquet --drift_start_day 0

echo "==> Drift traffic"
python src/demo/publish_and_score.py --url "${DEMO_SCORE_URL}" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_DRIFT}" --mode drift --seed "${DEMO_SEED}"

echo "==> Label + performance"
python src/feedback/run_feedback_loop.py --delay_minutes "${DEMO_DELAY_MINUTES}" --batch_size "${DEMO_BATCH_SIZE}"

echo "==> Monitoring snapshot (post drift)"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Open these URLs"
echo "Serving UI: ${DEMO_UI_URL}"
echo "Reports: ${DEMO_UI_URL%/ui}/reports/index.html"
