from __future__ import annotations

import json
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.request import Request, urlopen

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram, generate_latest
from starlette.responses import FileResponse, Response
from starlette.staticfiles import StaticFiles

ROUTER_MODE = os.getenv("ROUTER_MODE", "shadow").lower()
CANARY_PERCENT = int(os.getenv("CANARY_PERCENT", "10"))
V1_URL = os.getenv("V1_URL", "http://serving-v1:8080/score")
V2_URL = os.getenv("V2_URL", "http://serving-v2:8080/score")
UI_ROOT = os.path.join(os.path.dirname(__file__), "ui")

REQUESTS = PromCounter("router_requests_total", "Router requests")
CANARY_ROUTED = PromCounter("router_canary_routed_total", "Canary routed", ["to"])
SHADOW_CALLS = PromCounter("router_shadow_total", "Shadow calls")
SHADOW_ERRORS = PromCounter("router_shadow_errors_total", "Shadow errors")
SHADOW_DECISION_DIFF = PromCounter("router_shadow_decision_diff_total", "Shadow decision diffs")
SHADOW_SCORE_DELTA = Histogram("router_shadow_score_delta", "Shadow score delta")

app = FastAPI(title="Fraud Router")
app.mount("/static", StaticFiles(directory=UI_ROOT), name="static")

decisions = deque(maxlen=200)
shadow_comparisons = deque(maxlen=200)
router_counts = Counter()


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


def _post_json(url: str, payload: Dict[str, Any], timeout: float = 0.2) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _get_json(url: str, timeout: float = 0.2) -> Dict[str, Any]:
    req = Request(url)
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def _choose_canary(user_id: int) -> bool:
    return (hash(user_id) % 100) < CANARY_PERCENT


@app.get("/api/canary-preview")
def canary_preview(user_id: int):
    return {
        "user_id": user_id,
        "canary_percent": CANARY_PERCENT,
        "route_to": "v2" if _choose_canary(user_id) else "v1",
    }


@app.post("/score")
def score(event: TransactionEvent):
    REQUESTS.inc()
    router_counts["requests"] += 1
    payload = event.dict()
    use_v2 = _choose_canary(event.user_id)

    if ROUTER_MODE == "canary":
        target = V2_URL if use_v2 else V1_URL
        CANARY_ROUTED.labels(to="v2" if use_v2 else "v1").inc()
        response = _post_json(target, payload)
    else:
        response = _post_json(V1_URL, payload)
        CANARY_ROUTED.labels(to="v1").inc()

        def _shadow():
            router_counts["shadow_calls"] += 1
            SHADOW_CALLS.inc()
            try:
                v2_resp = _post_json(V2_URL, payload, timeout=0.15)
                v1_score = response.get("final_score") or 0.0
                v2_score = v2_resp.get("final_score") or 0.0
                score_delta = float(v2_score) - float(v1_score)
                decision_diff = v2_resp.get("decision") != response.get("decision")
                SHADOW_SCORE_DELTA.observe(score_delta)
                if decision_diff:
                    SHADOW_DECISION_DIFF.inc()
                shadow_comparisons.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "user_id": payload.get("user_id"),
                        "v1_score": v1_score,
                        "v2_score": v2_score,
                        "score_delta": score_delta,
                        "decision_diff": decision_diff,
                        "v1_decision": response.get("decision"),
                        "v2_decision": v2_resp.get("decision"),
                        "v1_latency": response.get("latency_ms", {}).get("total"),
                        "v2_latency": v2_resp.get("latency_ms", {}).get("total"),
                    }
                )
            except Exception as exc:
                router_counts["shadow_errors"] += 1
                SHADOW_ERRORS.inc()
                shadow_comparisons.append(
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "user_id": payload.get("user_id"),
                        "v2_error": str(exc),
                    }
                )

        ThreadPoolExecutor(max_workers=1).submit(_shadow)

    decisions.append(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_id": payload.get("event_id"),
            "user_id": payload.get("user_id"),
            "amount": payload.get("amount"),
            "country": payload.get("country"),
            "decision": response.get("decision"),
            "final_score": response.get("final_score"),
            "scores": response.get("scores"),
            "fallbacks": [key for key, value in response.get("fallbacks", {}).items() if value],
            "latency_ms": response.get("latency_ms", {}).get("total"),
            "served_by": response.get("served_by"),
        }
    )
    return response


@app.get("/ui")
def ui():
    return FileResponse(os.path.join(UI_ROOT, "index.html"))


@app.get("/api/stats")
def stats():
    def _safe_health(url: str) -> Dict[str, Any]:
        try:
            return _get_json(url.replace("/score", "/health"))
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    v1 = _safe_health(V1_URL)
    v2 = _safe_health(V2_URL)
    return {
        "service": "fraud-router",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "router": {
            "mode": ROUTER_MODE,
            "canary_percent": CANARY_PERCENT,
        },
        "v1": v1,
        "v2": v2,
        "counters": {
            "requests_total": int(router_counts.get("requests", 0)),
            "shadow_calls": int(router_counts.get("shadow_calls", 0)),
            "shadow_errors": int(router_counts.get("shadow_errors", 0)),
        },
    }


@app.get("/api/recent-decisions")
def recent_decisions(limit: int = 50):
    limit = max(1, min(limit, 200))
    return list(decisions)[-limit:]


@app.get("/api/shadow-comparisons")
def shadow(limit: int = 50):
    limit = max(1, min(limit, 200))
    return list(shadow_comparisons)[-limit:]


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
