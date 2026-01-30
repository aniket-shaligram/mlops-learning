#!/usr/bin/env bash
set -euo pipefail

echo "==> Get online features"
python - <<'PY'
import os
import sys
import psycopg2
from feast import FeatureStore

repo_root = os.getcwd()
sys.path.append(repo_root)

from feast_client import FEAST_FEATURE_REFS

conn = psycopg2.connect(os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc"))
cur = conn.cursor()
cur.execute("select user_id, merchant_id, device_id, ip_id from txn_validated order by event_ts desc limit 1")
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
