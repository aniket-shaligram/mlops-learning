from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from utils import FEATURES, load_json, make_feature_frame, validate_input_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single transaction.")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / "model.pkl"
    features_path = artifacts_dir / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features list not found at {features_path}")

    payload: Dict[str, Any] = load_json(args.input_json)
    validate_input_payload(payload)

    model = joblib.load(model_path)
    features_payload = load_json(features_path)
    feature_list = features_payload.get("features", FEATURES)

    row = {feature: payload[feature] for feature in feature_list}
    df = make_feature_frame(row)
    proba = float(model.predict_proba(df)[:, 1][0])

    decision = "fraud" if proba >= args.threshold else "legit"
    output = {
        "fraud_probability": proba,
        "threshold": args.threshold,
        "decision": decision,
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
