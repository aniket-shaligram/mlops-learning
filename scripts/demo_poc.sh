#!/usr/bin/env bash
set -euo pipefail

DEMO_QPS="${DEMO_QPS:-5}"
DEMO_SECONDS_BASELINE="${DEMO_SECONDS_BASELINE:-30}"
DEMO_SECONDS_DRIFT="${DEMO_SECONDS_DRIFT:-30}"
DEMO_SECONDS_CANARY="${DEMO_SECONDS_CANARY:-120}"
DEMO_DELAY_MINUTES="${DEMO_DELAY_MINUTES:-5}"
DEMO_BATCH_SIZE="${DEMO_BATCH_SIZE:-500}"
DEMO_SEED="${DEMO_SEED:-42}"
DEMO_ROUTER_URL="${DEMO_ROUTER_URL:-http://localhost:8080}"
DEMO_FEAST_REPO="${DEMO_FEAST_REPO:-feast_repo}"

echo "==> Starting infra"
docker compose -f ops/docker-compose.yml up -d

echo "==> Waiting for router"
python src/demo/wait_for_http.py --url "${DEMO_ROUTER_URL}/api/stats" --timeout 120

echo "==> Start GX worker"
mkdir -p logs
python src/gx/validate_and_forward.py > logs/gx_worker.log 2>&1 &
sleep 3

echo "==> Generate baseline synthetic data"
python src/synth/generate_synth.py --rows 200000 --format parquet --drift_start_day 9999

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

echo "==> Baseline traffic"
python src/demo/publish_and_score.py --url "${DEMO_ROUTER_URL}/score" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_BASELINE}" --mode baseline --seed "${DEMO_SEED}"

echo "==> Monitoring snapshot"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Drift dataset"
python src/synth/generate_synth.py --rows 200000 --format parquet --drift_start_day 0

echo "==> Drift traffic"
python src/demo/publish_and_score.py --url "${DEMO_ROUTER_URL}/score" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_DRIFT}" --mode drift --seed "${DEMO_SEED}"

echo "==> Label + performance"
python src/feedback/run_feedback_loop.py --delay_minutes "${DEMO_DELAY_MINUTES}" --batch_size "${DEMO_BATCH_SIZE}"

echo "==> Monitoring snapshot (post drift)"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Retrain v2 candidate"
python src/feedback/retrain_trigger.py --min_labeled_rows 200 --perf_floor_f1 0.6 --perf_floor_precision 0.5 || true
if [ -d artifacts_phase4_candidate ]; then
  CANDIDATE_DIR=$(ls -t artifacts_phase4_candidate | head -1 || true)
  if [ -n "${CANDIDATE_DIR}" ]; then
    python src/feedback/promote_candidate.py --candidate_dir "artifacts_phase4_candidate/${CANDIDATE_DIR}" --target v2 || true
  fi
fi

echo "==> Enable canary and send traffic"
python src/demo/try_enable_canary.py --base_url "${DEMO_ROUTER_URL}" --canary_percent 20 || true
python src/demo/publish_and_score.py --url "${DEMO_ROUTER_URL}/score" --qps "${DEMO_QPS}" --seconds "${DEMO_SECONDS_CANARY}" --mode canary --seed "${DEMO_SEED}"

echo "==> Compare models"
python src/demo/compare_models.py --minutes 30

echo "==> Final monitoring snapshot"
python src/monitoring/run_all.py --ref_hours 168 --cur_hours 24 || true

echo "==> Open these URLs"
echo "Router UI: ${DEMO_ROUTER_URL}/ui"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000"
echo "Reports: ${DEMO_ROUTER_URL}/reports/index.html"
