from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import psycopg2

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.serving.model_loader import load_models, _load_joblib, _resolve_local_champion_path
from src.utils import ensure_dir, save_json


def _pg_dsn() -> str:
    return os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")


def _parse_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(timezone.utc)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse datetime: {value}")
    return parsed.to_pydatetime()


def _window(as_of: datetime, ref_hours: int, cur_hours: int) -> Tuple[Tuple[datetime, datetime], Tuple[datetime, datetime]]:
    cur_end = as_of
    cur_start = as_of - timedelta(hours=cur_hours)
    ref_end = cur_start
    ref_start = ref_end - timedelta(hours=ref_hours)
    return (ref_start, ref_end), (cur_start, cur_end)


def _fetch_features(start: datetime, end: datetime, sample: int) -> pd.DataFrame:
    sql = """
        select features
        from decision_log
        where created_at >= %s and created_at < %s
          and features != '{}'::jsonb
        order by created_at desc
        limit %s
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        df = pd.read_sql(sql, conn, params=(start, end, sample))
    if df.empty:
        return df
    return pd.json_normalize(df["features"].apply(lambda x: x or {}))


def _prepare_frame(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    if not feature_list:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return df
    for col in feature_list:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_list]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def run(ref_hours: int = 24, cur_hours: int = 1, as_of: str = "now", sample_size: int = 500) -> None:
    as_of_dt = _parse_time(as_of)
    (ref_start, ref_end), (cur_start, cur_end) = _window(as_of_dt, ref_hours, cur_hours)

    report_dir = ensure_dir(
        Path("monitoring")
        / "reports"
        / "shap"
        / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )

    bundle = load_models()
    model = bundle.champion_model
    feature_list = bundle.metadata.get("feature_list", [])

    summary: Dict[str, Any] = {
        "reference_window": {"start": ref_start.isoformat(), "end": ref_end.isoformat()},
        "current_window": {"start": cur_start.isoformat(), "end": cur_end.isoformat()},
        "status": "ok",
    }

    if model is None:
        summary["status"] = "model_missing"
        save_json(report_dir / "summary.json", summary)
        return
    if model.__class__.__name__ == "PyFuncModel":
        champion_type = bundle.metadata.get("champion_type")
        registered_name = bundle.metadata.get("registered_model_name")
        local_path = _resolve_local_champion_path(champion_type, registered_name) if champion_type else None
        local_model = _load_joblib(local_path) if local_path else None
        if local_model is not None:
            model = local_model

    ref_df = _fetch_features(ref_start, ref_end, sample_size)
    cur_df = _fetch_features(cur_start, cur_end, sample_size)

    if ref_df.empty or cur_df.empty:
        if ref_df.empty and len(cur_df) >= 200:
            split_idx = int(len(cur_df) * 0.6)
            ref_df = cur_df.iloc[:split_idx].copy()
            cur_df = cur_df.iloc[split_idx:].copy()
            summary["status"] = "fallback_split"
            summary["ref_rows"] = int(len(ref_df))
            summary["cur_rows"] = int(len(cur_df))
        else:
            summary["status"] = "insufficient_data"
            summary["ref_rows"] = int(len(ref_df))
            summary["cur_rows"] = int(len(cur_df))
            save_json(report_dir / "summary.json", summary)
            return

    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as exc:
        summary["status"] = "missing_dependency"
        summary["error"] = str(exc)
        save_json(report_dir / "summary.json", summary)
        return

    ref_df = _prepare_frame(ref_df, feature_list)
    cur_df = _prepare_frame(cur_df, feature_list)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(cur_df)
    except Exception as exc:
        summary["status"] = "unsupported_model"
        summary["error"] = str(exc)
        save_json(report_dir / "summary.json", summary)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_abs = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(cur_df.columns.tolist(), shap_abs.tolist()))
    save_json(report_dir / "shap_importance.json", importance)

    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    pd.DataFrame(top_features, columns=["feature", "mean_abs_shap"]).to_csv(
        report_dir / "top_features.csv", index=False
    )

    plt.figure()
    shap.summary_plot(shap_values, cur_df, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_summary.png", dpi=150)

    # importance drift
    ref_values = explainer.shap_values(ref_df)
    if isinstance(ref_values, list):
        ref_values = ref_values[1] if len(ref_values) > 1 else ref_values[0]
    ref_abs = np.abs(ref_values).mean(axis=0)
    drift = {}
    for idx, col in enumerate(cur_df.columns):
        drift[col] = float(shap_abs[idx] - ref_abs[idx])
    save_json(report_dir / "importance_drift.json", drift)

    save_json(report_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SHAP explainability.")
    parser.add_argument("--ref_hours", type=int, default=24)
    parser.add_argument("--cur_hours", type=int, default=1)
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--sample_size", type=int, default=500)
    args = parser.parse_args()
    run(
        ref_hours=args.ref_hours,
        cur_hours=args.cur_hours,
        as_of=args.as_of,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
