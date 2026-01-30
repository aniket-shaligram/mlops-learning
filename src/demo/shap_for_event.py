from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd


def _load_features(path: str) -> Dict[str, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data


def _load_feature_order(path: str) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("features", [])


def _build_frame(features: Dict[str, float], order: List[str]) -> pd.DataFrame:
    row = {name: float(features.get(name, 0)) for name in order}
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Local SHAP for one event.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--features_json", required=True)
    parser.add_argument("--feature_order_json", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    features = _load_features(args.features_json)
    feature_order = _load_feature_order(args.feature_order_json)
    if not feature_order:
        raise SystemExit("feature_order.json missing or empty")

    X = _build_frame(features, feature_order)

    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    values = shap_values[0]

    rows = []
    for idx, name in enumerate(feature_order):
        rows.append(
            {
                "feature": name,
                "contribution": float(values[idx]),
                "abs_contribution": float(abs(values[idx])),
            }
        )
    rows.sort(key=lambda x: x["abs_contribution"], reverse=True)
    rows = rows[: args.top_k]

    output = {"top_contributors": rows}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
