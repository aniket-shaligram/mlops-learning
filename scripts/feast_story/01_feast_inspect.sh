#!/usr/bin/env bash
set -euo pipefail

echo "==> Feast inspect"
cd feast_repo
feast apply
feast entities list || true
feast feature-views list || true
feast feature-services list || true
