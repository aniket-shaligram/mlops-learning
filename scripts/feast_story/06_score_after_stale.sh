#!/usr/bin/env bash
set -euo pipefail

echo "==> Score after stale features"
python src/demo/publish_and_score.py --url http://localhost:8080/score --qps 5 --seconds 4 --mode drift --seed 42

docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c \
"select decision, count(*) from decision_log where created_at > now() - interval '10 minutes' group by decision;"
