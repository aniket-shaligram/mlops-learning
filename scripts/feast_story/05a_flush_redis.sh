#!/usr/bin/env bash
set -euo pipefail

echo "==> Flush Redis"
docker exec -i $(docker ps -qf "name=redis") redis-cli FLUSHALL
