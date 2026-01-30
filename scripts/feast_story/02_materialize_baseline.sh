#!/usr/bin/env bash
set -euo pipefail

echo "==> Materialize baseline"
python src/feast_materialize.py --start now-1d --end now
