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
