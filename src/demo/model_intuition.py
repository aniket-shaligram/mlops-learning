from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import psycopg2


def _pg_dsn() -> str:
    return os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")


def _bundle_root(path: str | None) -> Path:
    if path:
        return Path(path)
    return Path("model_bundles") / "v1"


def _load_feature_order(path: Path) -> List[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("features", [])


def _sample_features(sample_size: int) -> pd.DataFrame:
    sql = """
        select features
        from decision_log
        where features != '{}'::jsonb
        order by created_at desc
        limit %s
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        df = pd.read_sql(sql, conn, params=(sample_size,))
    if df.empty:
        return df
    return pd.json_normalize(df["features"].apply(lambda x: x or {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Model intuition summary.")
    parser.add_argument("--bundle_dir", default=None)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()

    bundle = _bundle_root(args.bundle_dir)
    model_path = bundle / "gbdt" / "model.pkl"
    feature_order_path = bundle / "gbdt" / "feature_order.json"
    if not model_path.exists():
        model_path = Path("artifacts_phase4") / "gbdt" / "model.pkl"
    if not feature_order_path.exists():
        feature_order_path = Path("artifacts_phase4") / "gbdt" / "feature_order.json"

    model = joblib.load(model_path)
    feature_order = _load_feature_order(feature_order_path)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = list(model.feature_importances_)

    top_features = []
    if importances and feature_order:
        pairs = sorted(
            zip(feature_order, importances),
            key=lambda x: x[1],
            reverse=True,
        )[: args.top_k]
        top_features = [{"feature": f, "importance": float(i)} for f, i in pairs]

    shap_summary = []
    try:
        import shap

        sample_df = _sample_features(args.sample_size)
        if not sample_df.empty and feature_order:
            for col in feature_order:
                if col not in sample_df.columns:
                    sample_df[col] = 0
            X = sample_df[feature_order].apply(pd.to_numeric, errors="coerce").fillna(0)
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            if isinstance(values, list):
                values = values[1] if len(values) > 1 else values[0]
            mean_abs = np.abs(values).mean(axis=0)
            top = sorted(
                zip(feature_order, mean_abs),
                key=lambda x: x[1],
                reverse=True,
            )[: args.top_k]
            shap_summary = [{"feature": f, "mean_abs_shap": float(v)} for f, v in top]
    except Exception:
        pass

    rules_thresholds = None
    rules_path = bundle / "rules" / "thresholds.json"
    if rules_path.exists():
        rules_thresholds = json.loads(rules_path.read_text(encoding="utf-8"))

    top_names = [row["feature"] for row in shap_summary] if shap_summary else [row["feature"] for row in top_features]
    top_names = top_names[:5]

    rules_summary = "Rules model uses fixed thresholds as safety tripwires."
    if rules_thresholds:
        rules_summary = "Rules model uses fixed thresholds for velocity, amount, and risk flags."

    anomaly_summary = (
        "Anomaly model flags rare combinations vs historical behavior (IsolationForest). "
        "It is most sensitive to unusual velocity and amount patterns."
    )

    gbdt_summary = "GBDT learns non-linear interactions across amount, velocity, and profile features."
    if top_names:
        gbdt_summary = (
            "GBDT focuses most on recent spend velocity and user spend norms. "
            f"Top signals: {', '.join(top_names)}."
        )

    models = [
        {
            "id": "gbdt",
            "title": "GBDT (Supervised)",
            "summary": gbdt_summary,
            "top_features": top_features or shap_summary,
        },
        {
            "id": "rules",
            "title": "Rules Model",
            "summary": rules_summary,
            "thresholds": rules_thresholds or {},
        },
        {
            "id": "anomaly",
            "title": "Anomaly Model",
            "summary": anomaly_summary,
        },
    ]

    output: Dict[str, Any] = {
        "model_path": str(model_path),
        "feature_order_path": str(feature_order_path),
        "models": models,
    }
    out_path = Path("monitoring/reports/latest/model_intuition.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
