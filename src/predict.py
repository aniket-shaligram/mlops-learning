from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from utils import encode_synth_features, get_dataset_config, load_json, make_feature_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a single transaction.")
    parser.add_argument("--artifacts_dir", default="artifacts")
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--dataset_type",
        choices=["kaggle", "synth"],
        default=None,
        help="Override dataset type from artifacts/run_config.json",
    )
    parser.add_argument("--use_feast", choices=["true", "false"], default=None)
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_path = artifacts_dir / "model.pkl"
    features_path = artifacts_dir / "features.json"
    run_config_path = artifacts_dir / "run_config.json"

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

    dataset_type = "kaggle"
    categorical_cols = []
    if run_config_path.exists():
        run_config = load_json(run_config_path)
        dataset_type = run_config.get("dataset_type", "kaggle")
        categorical_cols = run_config.get("categorical_cols", [])
    if args.dataset_type:
        dataset_type = args.dataset_type

    use_feast = args.use_feast
    if use_feast is None:
        use_feast = dataset_type == "synth"
    else:
        use_feast = use_feast.lower() == "true"

    if dataset_type == "synth":
        raw_features, _, synth_categoricals = get_dataset_config("synth")
        categorical_cols = categorical_cols or synth_categoricals
        if use_feast:
            from feast_client import get_online_feature_vector

            feast_features = get_online_feature_vector(payload)
            merged_payload = dict(payload)
            merged_payload.update(feast_features)
            df_raw = make_feature_frame(merged_payload, raw_features)
        else:
            df_raw = make_feature_frame(payload, raw_features)
        df = encode_synth_features(df_raw, categorical_cols, feature_list).astype(float)
    else:
        if use_feast:
            raise ValueError("Feast online features are only supported for synth.")
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
