#!/usr/bin/env bash
set -euo pipefail

echo "==> Score high-risk transaction"
curl -s -X POST http://localhost:8080/score \
  -H 'Content-Type: application/json' \
  -d '{"event_id":"demo_evt_hr","event_type":"txn.created","event_ts":"2026-01-29T00:00:00Z","user_id":1,"merchant_id":10,"device_id":5,"ip_id":7,"amount":4500.5,"currency":"USD","country":"NG","channel":"mobile","drift_phase":1}'

echo
echo "==> Latest decision feature snapshot"
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c \
"select event_id, final_score, decision, features from decision_log order by created_at desc limit 1;"
