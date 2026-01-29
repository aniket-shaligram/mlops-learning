from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from utils import load_json, make_feature_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single transaction.")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / "model.pkl"
    features_path = artifacts_dir / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features list not found at {features_path}")

    payload: Dict[str, Any] = load_json(args.input_json)

    model = joblib.load(model_path)
    features_payload = load_json(features_path)
    feature_list = features_payload.get("features", [])
    if not feature_list:
        raise ValueError("Features list is empty in artifacts/features.json")

    df = make_feature_frame(payload, feature_list).astype(float)
    proba = float(model.predict_proba(df)[:, 1][0])

    threshold = args.threshold
    if threshold is None:
        metrics_path = artifacts_dir / "metrics.json"
        if metrics_path.exists():
            metrics_payload = load_json(metrics_path)
            threshold = metrics_payload.get("optimal_threshold")
        if threshold is None:
            threshold = 0.5

    decision = "fraud" if proba >= threshold else "legit"
    output = {
        "fraud_probability": proba,
        "threshold": threshold,
        "decision": decision,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
