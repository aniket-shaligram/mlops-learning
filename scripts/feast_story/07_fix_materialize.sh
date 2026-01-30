#!/usr/bin/env bash
set -euo pipefail

echo "==> Re-materialize and check features"
python src/feast_materialize.py --start now-1d --end now
./scripts/feast_story/03_get_online_features.sh
./scripts/feast_story/04_score_and_show_features.sh
