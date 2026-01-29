from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

from src.serving.feast_client import OnlineFeatureClient
from src.serving.model_loader import load_models
from src.serving.scorer import score_event

LOG_FULL_PAYLOAD = os.getenv("LOG_FULL_PAYLOAD", "false").lower() == "true"

logger = logging.getLogger("serving")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
ERRORS = Counter("errors_total", "Total errors", ["endpoint"])
FALLBACKS = Counter("fallbacks_total", "Fallback events", ["type"])
FEAST_FAILURES = Counter("feast_failures_total", "Feast fetch failures")
MODEL_FAILURES = Counter("model_failures_total", "Model failures", ["model"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])


class TransactionEvent(BaseModel):
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    event_ts: Optional[str] = None
    user_id: int
    merchant_id: int
    device_id: int
    ip_id: int
    amount: float
    currency: str
    country: str
    channel: str
    drift_phase: int


app = FastAPI(title="Fraud Inference Gateway")
models = load_models()
feast_client = OnlineFeatureClient()


def _log_decision(event: Dict[str, Any], response: Dict[str, Any], feature_count: int) -> None:
    payload = {
        "event_id": event.get("event_id"),
        "user_id": event.get("user_id"),
        "merchant_id": event.get("merchant_id"),
        "device_id": event.get("device_id"),
        "decision": response.get("decision"),
        "final_score": response.get("final_score"),
        "scores": response.get("scores"),
        "fallbacks": response.get("fallbacks"),
        "feature_count": feature_count,
        "latency_ms": response.get("latency_ms", {}).get("total"),
    }
    if LOG_FULL_PAYLOAD:
        payload["event"] = event
    logger.info(json.dumps(payload))


@app.post("/score")
def score(event: TransactionEvent):
    start = time.perf_counter()
    REQUESTS.labels(endpoint="/score").inc()

    event_dict = event.dict()
    features, feast_failed = feast_client.fetch_online_features(event_dict)
    if feast_failed:
        FEAST_FAILURES.inc()

    try:
        response, _latencies = score_event(
            event_dict,
            features,
            {
                "rules": models.rules_model,
                "champion": models.champion_model,
                "anomaly": models.anomaly_model,
            },
            feature_list=models.metadata.get("feature_list") or [],
        )
    except Exception:
        ERRORS.labels(endpoint="/score").inc()
        raise

    for name, value in response.get("fallbacks", {}).items():
        if value:
            FALLBACKS.labels(type=name).inc()
            if name == "champion_missing":
                MODEL_FAILURES.labels(model="champion").inc()
            if name == "anomaly_missing":
                MODEL_FAILURES.labels(model="anomaly").inc()

    response["model_versions"] = {
        "champion_type": models.metadata.get("champion_type"),
        "registry_mode": models.metadata.get("registry_mode"),
    }

    _log_decision(event_dict, response, feature_count=len(features))
    LATENCY.labels(endpoint="/score").observe(time.perf_counter() - start)
    return response


@app.get("/health")
def health():
    feast_ok, feast_error = feast_client.health_check()
    return {
        "status": "ok",
        "models": {
            "rules": True,
            "champion": models.champion_model is not None,
            "anomaly": models.anomaly_model is not None,
        },
        "feast": {"ok": feast_ok, "error": feast_error},
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
