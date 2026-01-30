# Fraud Scoring Baseline (Synthetic)

Minimal, production-ish baseline for scoring fraud using an interpretable synthetic dataset.
It trains models locally, saves artifacts, and scores transactions via a FastAPI gateway.

Docs:
- `README_SHORT.md` — minimal end-to-end run
- `README_DETAILED.md` — full step-by-step guide

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you’re new to the repo, start with `README_SHORT.md`. For full details
(Feast, MLflow, Serving, Monitoring, Canary/Shadow), use `README_DETAILED.md`.

## Quickstart demo
```bash
chmod +x scripts/demo_*.sh scripts/feast_story/*.sh
./scripts/demo_baseline.sh
./scripts/demo_drift.sh
./scripts/demo_retrain.sh
# optional canary story
./scripts/demo_canary.sh
```

## Feast story
Run in order:
```
scripts/feast_story/01_feast_inspect.sh
scripts/feast_story/02_materialize_baseline.sh
scripts/feast_story/03_get_online_features.sh
scripts/feast_story/04_score_and_show_features.sh
scripts/feast_story/05a_flush_redis.sh
scripts/feast_story/06_score_after_stale.sh
scripts/feast_story/07_fix_materialize.sh
```

## What to open
- Serving UI (v1): `http://localhost:8083/ui`
- Serving UI (v2): `http://localhost:8084/ui`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Reports: `http://localhost:8083/reports/index.html`

## How to know it worked
- GX quarantine: `txn_quarantine` table has rows
- Feast online: `scripts/feast_story/03_get_online_features.sh` prints JSON
- Serving ensemble: `/score` returns `scores` and `fallbacks`
- Canary/shadow: `./scripts/demo_canary.sh`
- Drift/SHAP/slices: `monitoring/reports/latest/`
- Feedback loop: `feedback/reports/retrain/trigger.json`
