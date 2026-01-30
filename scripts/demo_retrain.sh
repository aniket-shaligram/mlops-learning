#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_CANARY="${DEMO_SECONDS_CANARY:-120}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_SCORE_URL="${DEMO_SCORE_URL:-http://localhost:8083/score}"
DEMO_UI_URL="${DEMO_UI_URL:-http://localhost:8083/ui}"

echo "==> Retrain v2 candidate"
python src/feedback/retrain_trigger.py --min_labeled_rows 200 --perf_floor_f1 0.6 --perf_floor_precision 0.5 || true

if [ -d artifacts_phase4_candidate ]; then
  CANDIDATE_DIR=$(ls -t artifacts_phase4_candidate | head -1 || true)
  if [ -n "${CANDIDATE_DIR}" ]; then
    python src/feedback/promote_candidate.py --candidate_dir "artifacts_phase4_candidate/${CANDIDATE_DIR}" --target v2 || true
    docker compose -f ops/docker-compose.yml up -d --build serving-v2
    python src/demo/wait_for_http.py --url "http://localhost:8084/health" --timeout 120
  fi
fi

echo "==> Send traffic to v2"
python src/demo/publish_and_score.py --url "http://localhost:8084/score" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_CANARY}" --mode baseline --seed "${DEMO_SEED}"

echo "==> Compare models"
python src/demo/compare_models.py --minutes 30

echo "==> Final monitoring snapshot"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Open these URLs"
echo "Serving UI v1: ${DEMO_UI_URL}"
echo "Serving UI v2: http://localhost:8084/ui"
echo "Reports: ${DEMO_UI_URL%/ui}/reports/index.html"
