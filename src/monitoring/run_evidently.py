from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import psycopg2

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

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


def _fetch_decisions(start: datetime, end: datetime) -> pd.DataFrame:
    sql = """
        select event_ts, amount, country, channel, drift_phase, final_score, decision, features
        from decision_log
        where created_at >= %s and created_at < %s
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        return pd.read_sql(sql, conn, params=(start, end))


def _flatten_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    features = df["features"].apply(lambda x: x or {})
    flat = pd.json_normalize(features)
    return flat


def _safe_dataset_drift(report_dict: Dict[str, Any]) -> bool:
    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "dataset_drift" in result:
            return bool(result["dataset_drift"])
    return False


def _top_drift_columns(report_dict: Dict[str, Any]) -> list[Dict[str, Any]]:
    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "drift_by_columns" in result:
            items = []
            for col, info in result["drift_by_columns"].items():
                items.append({"column": col, "drift_score": info.get("drift_score")})
            items = [item for item in items if item["drift_score"] is not None]
            return sorted(items, key=lambda x: x["drift_score"], reverse=True)[:5]
    return []


def run(ref_hours: int = 24, cur_hours: int = 1, as_of: str = "now") -> None:
    as_of_dt = _parse_time(as_of)
    (ref_start, ref_end), (cur_start, cur_end) = _window(as_of_dt, ref_hours, cur_hours)

    ref_df = _fetch_decisions(ref_start, ref_end)
    cur_df = _fetch_decisions(cur_start, cur_end)

    report_dir = ensure_dir(
        Path("monitoring")
        / "reports"
        / "evidently"
        / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )

    summary: Dict[str, Any] = {
        "reference_window": {"start": ref_start.isoformat(), "end": ref_end.isoformat()},
        "current_window": {"start": cur_start.isoformat(), "end": cur_end.isoformat()},
        "status": "ok",
    }

    if len(ref_df) < 200 or len(cur_df) < 200:
        total_rows = int(len(ref_df) + len(cur_df))
        if len(ref_df) == 0 and len(cur_df) >= 200:
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
            summary["total_rows"] = total_rows
            save_json(report_dir / "summary.json", summary)
            return

    metric_ctor = None
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        metric_ctor = DataDriftPreset
    except Exception:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset
            metric_ctor = DataDriftPreset
        except Exception as exc:
            summary["status"] = "missing_dependency"
            summary["error"] = str(exc)
            save_json(report_dir / "summary.json", summary)
            return

    input_cols = ["amount", "country", "channel", "drift_phase"]
    ref_inputs = ref_df[input_cols].copy()
    cur_inputs = cur_df[input_cols].copy()

    input_report = Report(metrics=[metric_ctor()])
    input_snapshot = input_report.run(current_data=cur_inputs, reference_data=ref_inputs)
    input_snapshot.save_json(str(report_dir / "input_drift.json"))
    input_snapshot.save_html(str(report_dir / "input_drift.html"))

    feature_ref = _flatten_features(ref_df)
    feature_cur = _flatten_features(cur_df)
    feature_report = Report(metrics=[metric_ctor()])
    feature_snapshot = feature_report.run(current_data=feature_cur, reference_data=feature_ref)
    feature_snapshot.save_json(str(report_dir / "feature_drift.json"))
    feature_snapshot.save_html(str(report_dir / "feature_drift.html"))

    pred_ref = ref_df[["final_score", "decision"]].copy()
    pred_cur = cur_df[["final_score", "decision"]].copy()
    pred_report = Report(metrics=[metric_ctor()])
    pred_snapshot = pred_report.run(current_data=pred_cur, reference_data=pred_ref)
    pred_snapshot.save_json(str(report_dir / "prediction_drift.json"))
    pred_snapshot.save_html(str(report_dir / "prediction_drift.html"))

    summary["input_drift"] = {
        "drift_detected": _safe_dataset_drift(input_snapshot.dict()),
        "top_columns": _top_drift_columns(input_snapshot.dict()),
    }
    summary["feature_drift"] = {
        "drift_detected": _safe_dataset_drift(feature_snapshot.dict()),
        "top_columns": _top_drift_columns(feature_snapshot.dict()),
    }
    summary["prediction_drift"] = {
        "drift_detected": _safe_dataset_drift(pred_snapshot.dict()),
        "top_columns": _top_drift_columns(pred_snapshot.dict()),
    }

    save_json(report_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Evidently drift reports.")
    parser.add_argument("--ref_hours", type=int, default=24)
    parser.add_argument("--cur_hours", type=int, default=1)
    parser.add_argument("--as_of", default="now")
    args = parser.parse_args()
    run(ref_hours=args.ref_hours, cur_hours=args.cur_hours, as_of=args.as_of)


if __name__ == "__main__":
    main()
