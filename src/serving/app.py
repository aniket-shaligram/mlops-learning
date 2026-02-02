from __future__ import annotations

import json
import logging
import os
import socket
import time
from collections import Counter as PyCounter
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extras import Json
from fastapi import FastAPI, Request
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
from starlette.responses import FileResponse
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from typing import List
import pandas as pd
import numpy as np
import shap

from src.serving.feast_client import OnlineFeatureClient
from src.serving.model_loader import load_models
from src.serving.scorer import score_event
from src.models.rules_model import RulesModel
from src.utils import encode_synth_features, SYNTH_CATEGORICALS

LOG_FULL_PAYLOAD = os.getenv("LOG_FULL_PAYLOAD", "false").lower() == "true"
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
UI_ROOT = os.path.join(os.path.dirname(__file__), "ui")
PG_DSN = os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")

logger = logging.getLogger("serving")
logging.basicConfig(level=logging.INFO, format="%(message)s")

REQUESTS = Counter("requests_total", "Total requests", ["endpoint"])
ERRORS = Counter("errors_total", "Total errors", ["endpoint"])
FALLBACKS = Counter("fallbacks_total", "Fallback events", ["type"])
FEAST_FAILURES = Counter("feast_failures_total", "Feast fetch failures")
MODEL_FAILURES = Counter("model_failures_total", "Model failures", ["model"])
LATENCY = Histogram("request_latency_seconds", "Request latency", ["endpoint"])

decisions = deque(maxlen=200)
request_counts = PyCounter()
error_counts = PyCounter()
fallback_counts = PyCounter()
feast_failure_count = 0


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
app.mount("/static", StaticFiles(directory=UI_ROOT), name="static")
app.mount("/reports", StaticFiles(directory="monitoring/reports"), name="reports")


@app.middleware("http")
async def no_cache_for_ui(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/ui" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


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


def _insert_decision_log(event: Dict[str, Any], response: Dict[str, Any]) -> None:
    event_ts = event.get("event_ts") or datetime.now(timezone.utc).isoformat()
    payload = {
        "event_id": event.get("event_id"),
        "event_ts": event_ts,
        "user_id": event.get("user_id"),
        "merchant_id": event.get("merchant_id"),
        "device_id": event.get("device_id"),
        "ip_id": event.get("ip_id"),
        "amount": event.get("amount"),
        "country": event.get("country"),
        "channel": event.get("channel"),
        "drift_phase": event.get("drift_phase", 0),
        "final_score": response.get("final_score", 0.0),
        "decision": response.get("decision", "unknown"),
        "scores": response.get("scores", {}),
        "fallbacks": response.get("fallbacks", {}),
        "model_versions": response.get("model_versions", {}),
        "features": response.get("feature_snapshot", {}),
        "latency_ms": response.get("latency_ms", {}),
        "served_by": response.get("served_by") or MODEL_VERSION or "v1",
    }
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                insert into decision_log (
                    event_id, event_ts, user_id, merchant_id, device_id, ip_id,
                    amount, country, channel, drift_phase, final_score, decision,
                    scores, fallbacks, model_versions, features, latency_ms, served_by
                ) values (
                    %(event_id)s, %(event_ts)s, %(user_id)s, %(merchant_id)s, %(device_id)s, %(ip_id)s,
                    %(amount)s, %(country)s, %(channel)s, %(drift_phase)s, %(final_score)s, %(decision)s,
                    %(scores)s, %(fallbacks)s, %(model_versions)s, %(features)s, %(latency_ms)s, %(served_by)s
                )
                """,
                {
                    **payload,
                    "scores": Json(payload["scores"]),
                    "fallbacks": Json(payload["fallbacks"]),
                    "model_versions": Json(payload["model_versions"]),
                    "features": Json(payload["features"]),
                    "latency_ms": Json(payload["latency_ms"]),
                },
            )


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _get_feature_value(features: Dict[str, Any], key: str) -> float:
    value = features.get(key, 0)
    if value is None:
        return 0.0
    return _to_float(value)


def _rules_triggered(features: Dict[str, Any], thresholds: Dict[str, float]) -> List[str]:
    amount = _get_feature_value(features, "amount")
    geo_mismatch = _get_feature_value(features, "geo_mismatch")
    is_new_device = _get_feature_value(features, "is_new_device")
    is_new_ip = _get_feature_value(features, "is_new_ip")
    suspicious_context = (geo_mismatch + is_new_device + is_new_ip) >= 1

    high_amount = amount >= thresholds.get("high_amount", 1500.0)
    very_high_amount = amount >= thresholds.get("very_high_amount", 5000.0)
    velocity_count = _get_feature_value(features, "user_txn_count_5m")
    velocity_amount = _get_feature_value(features, "user_amount_sum_1h")
    merchant_risk = _get_feature_value(features, "merchant_chargeback_rate_30d")

    triggered: List[str] = []
    if very_high_amount and suspicious_context:
        triggered.append("Very high amount + suspicious context")
    if high_amount and suspicious_context:
        triggered.append("High amount + suspicious context")
    if velocity_count >= thresholds.get("velocity_count_5m", 6.0):
        triggered.append("High transaction velocity (5m)")
    if velocity_amount >= thresholds.get("velocity_amount_1h", 2500.0):
        triggered.append("High spending velocity (1h)")
    if merchant_risk >= thresholds.get("merchant_risk", 0.8):
        triggered.append("High-risk merchant")
    return triggered


def _anomaly_reasons(anomaly_score: float, features: Dict[str, Any], thresholds: Dict[str, float]) -> List[str]:
    reasons: List[str] = []
    if anomaly_score > 0.8:
        reasons.append("Transaction looks highly unusual compared to baseline patterns")
    if _get_feature_value(features, "geo_mismatch") >= 1:
        reasons.append("Geo mismatch")
    if _get_feature_value(features, "is_new_device") >= 1:
        reasons.append("New device")
    if _get_feature_value(features, "is_new_ip") >= 1:
        reasons.append("New IP")
    velocity_count = _get_feature_value(features, "user_txn_count_5m")
    velocity_amount = _get_feature_value(features, "user_amount_sum_1h")
    if velocity_count >= thresholds.get("velocity_count_5m", 6.0) or velocity_amount >= thresholds.get(
        "velocity_amount_1h", 2500.0
    ):
        reasons.append("Unusual velocity")
    return reasons[:4]


def _build_feature_df(features: Dict[str, Any], feature_list: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{k: v for k, v in features.items() if v is not None}])
    if any(col in df.columns for col in SYNTH_CATEGORICALS):
        df = encode_synth_features(df, [c for c in SYNTH_CATEGORICALS if c in df.columns])
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]
    return df


def _compute_shap_top(
    features: Dict[str, Any], top_k: int, feature_list: list[str]
) -> List[Dict[str, Any]]:
    try:
        champion = models.champion_model
        if champion is None:
            return []
        df = _build_feature_df(features, feature_list)
        explainer = shap.TreeExplainer(champion)
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        values = shap_values[0] if isinstance(shap_values, np.ndarray) else np.array(shap_values)[0]
        pairs = list(zip(feature_list, values))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        output = []
        for name, val in pairs[: top_k or 5]:
            output.append(
                {
                    "feature": name,
                    "shap": float(val),
                    "direction": "increased_risk" if val > 0 else "decreased_risk",
                }
            )
        return output
    except Exception:
        return []


@app.post("/score")
def score(event: TransactionEvent):
    start = time.perf_counter()
    REQUESTS.labels(endpoint="/score").inc()
    request_counts["/score"] += 1

    event_dict = event.dict()
    features, feast_failed = feast_client.fetch_online_features(event_dict)
    if feast_failed:
        FEAST_FAILURES.inc()
        global feast_failure_count
        feast_failure_count += 1
        fallback_counts["feast_failed"] += 1

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
        error_counts["/score"] += 1
        raise

    response.setdefault("fallbacks", {})
    response["fallbacks"]["feast_failed"] = feast_failed

    for name, value in response.get("fallbacks", {}).items():
        if value:
            FALLBACKS.labels(type=name).inc()
            fallback_counts[name] += 1
            if name == "champion_missing":
                MODEL_FAILURES.labels(model="champion").inc()
            if name == "anomaly_missing":
                MODEL_FAILURES.labels(model="anomaly").inc()

    response["model_versions"] = {
        "champion_type": models.metadata.get("champion_type"),
        "registry_mode": models.metadata.get("registry_mode"),
        "champion_ref": (models.metadata.get("champion_uri") or "")[:40],
    }
    response["served_by"] = MODEL_VERSION
    response["feature_snapshot"] = features
    response["feast_failed"] = feast_failed

    _log_decision(event_dict, response, feature_count=len(features))
    try:
        _insert_decision_log(event_dict, response)
    except Exception as exc:
        logger.error(json.dumps({"decision_log_error": str(exc)}))
    decisions.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_id": event_dict.get("event_id"),
            "user_id": event_dict.get("user_id"),
            "amount": event_dict.get("amount"),
            "country": event_dict.get("country"),
            "decision": response.get("decision"),
            "final_score": response.get("final_score"),
            "scores": response.get("scores"),
            "fallbacks": [key for key, value in response.get("fallbacks", {}).items() if value],
            "latency_ms": response.get("latency_ms", {}).get("total"),
            "served_by": response.get("served_by"),
        }
    )
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
            "champion_ref": (models.metadata.get("champion_uri") or "")[:40],
        },
        "feast": {"ok": feast_ok, "error": feast_error},
        "served_by": MODEL_VERSION,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.get("/")
def root():
    return RedirectResponse(url="/ui")


@app.get("/ui")
def ui():
    return FileResponse(os.path.join(UI_ROOT, "index.html"))


def _redis_check() -> Dict[str, Any]:
    try:
        with socket.create_connection((REDIS_HOST, REDIS_PORT), timeout=0.3) as sock:
            sock.sendall(b"*1\r\n$4\r\nPING\r\n")
            data = sock.recv(16)
            if b"PONG" in data:
                return {"ok": True, "error": None}
            return {"ok": False, "error": "Unexpected PING response"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


@app.get("/api/stats")
def stats():
    feast_ok, feast_error = feast_client.health_check()
    redis_status = _redis_check()
    return {
        "service": "fraud-poc",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "served_by": MODEL_VERSION,
        "models": {
            "rules": True,
            "champion_loaded": models.champion_model is not None,
            "anomaly_loaded": models.anomaly_model is not None,
            "champion_type": models.metadata.get("champion_type"),
            "registry_mode": models.metadata.get("registry_mode", "none"),
            "champion_ref": (models.metadata.get("champion_uri") or "")[:40],
        },
        "feast": {"ok": feast_ok, "error": feast_error},
        "redis": redis_status,
        "counters": {
            "requests_total": int(request_counts.get("/score", 0)),
            "errors_total": int(error_counts.get("/score", 0)),
            "fallbacks_total": dict(fallback_counts),
            "feast_failed": int(feast_failure_count),
        },
    }


@app.get("/api/recent-decisions")
def recent_decisions(limit: int = 50):
    limit = max(1, min(limit, 200))
    return list(decisions)[-limit:]


@app.get("/api/txns/{event_id}/explain")
def explain_transaction(event_id: str, top_k: int = 5, include_features: bool = False):
    try:
        with psycopg2.connect(PG_DSN) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    select event_id, final_score, decision, scores, fallbacks, model_versions, features
                    from decision_log
                    where event_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (event_id,),
                )
                row = cursor.fetchone()
        if not row:
            return {"status": "error", "message": "event_id not found", "event_id": event_id}

        feature_snapshot = row[6] or {}
        scores = row[3] or {}
        fallbacks = row[4] or {}
        model_versions = row[5] or {}

        rules_model = models.rules_model if getattr(models, "rules_model", None) else RulesModel()
        thresholds = rules_model.thresholds if hasattr(rules_model, "thresholds") else RulesModel().thresholds
        rules_triggered = _rules_triggered(feature_snapshot, thresholds)

        feature_list = models.metadata.get("feature_list") or []
        gbdt_top = _compute_shap_top(feature_snapshot, top_k, feature_list) if feature_list else []

        anomaly_score = _to_float(scores.get("anomaly", 0.0))
        anomaly_reasons = _anomaly_reasons(anomaly_score, feature_snapshot, thresholds)

        payload: Dict[str, Any] = {
            "status": "ok",
            "event_id": row[0],
            "decision": row[2],
            "final_score": _to_float(row[1]),
            "scores": scores,
            "fallbacks": fallbacks,
            "model_versions": model_versions,
            "rules": {"thresholds": thresholds, "triggered": rules_triggered},
            "gbdt": {"top_contributors": gbdt_top},
            "anomaly": {"score": anomaly_score, "reasons": anomaly_reasons},
        }
        if include_features:
            payload["feature_snapshot"] = feature_snapshot
        return payload
    except Exception as exc:
        return {"status": "error", "message": str(exc), "event_id": event_id}


@app.get("/debug/last-decision")
def debug_last_decision():
    if not DEMO_MODE:
        return {"status": "disabled"}
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "select decision_id, event_id, event_ts, final_score, decision, served_by, model_versions, scores, features, created_at from decision_log order by created_at desc limit 1"
            )
            row = cursor.fetchone()
    if not row:
        return {"status": "empty"}
    return {
        "decision_id": row[0],
        "event_id": row[1],
        "event_ts": row[2].isoformat() if row[2] else None,
        "final_score": row[3],
        "decision": row[4],
        "served_by": row[5],
        "model_versions": row[6],
        "scores": row[7],
        "feature_snapshot": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
    }


@app.get("/debug/decision/{transaction_id}")
def debug_decision(transaction_id: str):
    if not DEMO_MODE:
        return {"status": "disabled"}
    with psycopg2.connect(PG_DSN) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                select decision_id, event_id, event_ts, final_score, decision, served_by, model_versions, scores, features, created_at
                from decision_log
                where event_id = %s
                order by created_at desc
                limit 1
                """,
                (transaction_id,),
            )
            row = cursor.fetchone()
    if not row:
        return {"status": "not_found"}
    return {
        "decision_id": row[0],
        "event_id": row[1],
        "event_ts": row[2].isoformat() if row[2] else None,
        "final_score": row[3],
        "decision": row[4],
        "served_by": row[5],
        "model_versions": row[6],
        "scores": row[7],
        "feature_snapshot": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
    }
