from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import SYNTH_CATEGORICALS, encode_synth_features


def _score_rules(model, df: pd.DataFrame) -> float:
    return float(model.predict_proba(df)[0][1])


def _score_champion(model, df: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        if proba.shape[1] == 1:
            return 0.0
        return float(proba[0][1])
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(df)[0])
        return max(0.0, min(1.0, score))
    pred = float(model.predict(df)[0])
    return max(0.0, min(1.0, pred))


def _score_anomaly(model, df: pd.DataFrame) -> float:
    return float(model.predict_proba(df)[0][1])


def _build_input_df(event: Dict[str, Any], features: Dict[str, Any], feature_list: list[str] | None) -> pd.DataFrame:
    payload: Dict[str, Any] = {
        "amount": event.get("amount", 0.0),
        "currency": event.get("currency", "USD"),
        "country": event.get("country", "US"),
        "channel": event.get("channel", "web"),
        "drift_phase": event.get("drift_phase", 0),
    }
    payload.update({k: v for k, v in features.items() if v is not None})
    df = pd.DataFrame([payload])
    if any(col in df.columns for col in SYNTH_CATEGORICALS):
        df = encode_synth_features(df, [c for c in SYNTH_CATEGORICALS if c in df.columns])
    if feature_list:
        for col in feature_list:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_list]
    return df


def score_event(
    event: Dict[str, Any],
    features: Dict[str, Any],
    models: Dict[str, Any],
    feature_list: list[str],
    timeout_ms: int = 50,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    start = time.perf_counter()
    df = _build_input_df(event, features, feature_list)

    scores: Dict[str, float] = {}
    fallbacks: Dict[str, bool] = {"rules_only": False, "champion_missing": False, "anomaly_missing": False}

    def _run(name: str):
        if name == "rules":
            return _score_rules(models["rules"], df)
        if name == "champion":
            if models.get("champion") is None:
                raise ValueError("champion_missing")
            return _score_champion(models["champion"], df)
        if name == "anomaly":
            if models.get("anomaly") is None:
                raise ValueError("anomaly_missing")
            return _score_anomaly(models["anomaly"], df)
        raise ValueError(f"Unknown scorer {name}")

    latencies: Dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            name: executor.submit(_run, name) for name in ["rules", "champion", "anomaly"]
        }
        for name, future in futures.items():
            t0 = time.perf_counter()
            try:
                scores[name] = float(future.result(timeout=timeout_ms / 1000))
            except TimeoutError:
                fallbacks[f"{name}_timeout"] = True
            except Exception:
                if name == "champion":
                    fallbacks["champion_missing"] = True
                if name == "anomaly":
                    fallbacks["anomaly_missing"] = True
            latencies[name] = (time.perf_counter() - t0) * 1000

    rules_score = scores.get("rules", 0.0)
    champion_score = scores.get("champion")
    anomaly_score = scores.get("anomaly")

    if champion_score is not None and anomaly_score is not None:
        blended = 0.6 * champion_score + 0.4 * anomaly_score
        final_score = max(rules_score, blended)
    elif champion_score is not None:
        blended = 0.8 * champion_score + 0.2 * rules_score
        final_score = max(rules_score, blended)
    elif anomaly_score is not None:
        blended = 0.7 * anomaly_score + 0.3 * rules_score
        final_score = max(rules_score, blended)
    else:
        fallbacks["rules_only"] = True
        final_score = rules_score

    if final_score >= 0.90:
        decision = "block"
    elif final_score >= 0.70:
        decision = "review"
    elif final_score >= 0.40:
        decision = "step_up"
    else:
        decision = "approve"

    total_latency = (time.perf_counter() - start) * 1000
    response = {
        "decision": decision,
        "final_score": float(final_score),
        "scores": {
            "rules": scores.get("rules"),
            "champion": scores.get("champion"),
            "anomaly": scores.get("anomaly"),
        },
        "fallbacks": fallbacks,
        "latency_ms": {**latencies, "total": total_latency},
    }
    return response, latencies
