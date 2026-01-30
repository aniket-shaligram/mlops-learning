from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import psycopg2
from sklearn.metrics import precision_recall_fscore_support

repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root))

from src.utils import ensure_dir, save_json


def _pg_dsn() -> str:
    return "postgresql://fraud:fraud@localhost:5432/fraud_poc"


def _parse_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(timezone.utc)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse datetime: {value}")
    return parsed.to_pydatetime()


def _window(as_of: datetime, hours: int) -> Tuple[datetime, datetime]:
    end = as_of
    start = as_of - timedelta(hours=hours)
    return start, end


def _fetch_decisions(start: datetime, end: datetime) -> pd.DataFrame:
    sql = """
        select event_id, final_score, decision, country, channel, drift_phase, user_id, merchant_id, device_id, event_ts
        from decision_log
        where created_at >= %s and created_at < %s
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        return pd.read_sql(sql, conn, params=(start, end))


def _fetch_labels(start: datetime, end: datetime) -> pd.DataFrame:
    sql = """
        select event_id, (payload->>'is_fraud')::int as label
        from txn_validated
        where event_ts >= %s and event_ts < %s
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        return pd.read_sql(sql, conn, params=(start, end))


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def run(hours: int = 24, as_of: str = "now") -> None:
    as_of_dt = _parse_time(as_of)
    start, end = _window(as_of_dt, hours)

    report_dir = ensure_dir(
        Path("monitoring")
        / "reports"
        / "slices"
        / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )

    decisions = _fetch_decisions(start, end)
    labels = _fetch_labels(start, end)

    if decisions.empty or labels.empty:
        save_json(report_dir / "summary.json", {"status": "insufficient_data"})
        return

    merged = decisions.merge(labels, on="event_id", how="left")
    if merged["label"].isna().all():
        save_json(report_dir / "summary.json", {"status": "labels_missing"})
        return

    merged["label"] = merged["label"].fillna(0).astype(int)
    merged["pred"] = (merged["final_score"] >= 0.5).astype(int)

    slices = []
    for col in ["country", "channel", "drift_phase"]:
        for value, group in merged.groupby(col):
            metrics = _metrics(group["label"], group["pred"])
            slices.append({"slice": col, "value": value, **metrics, "count": int(len(group))})

    slice_df = pd.DataFrame(slices)
    slice_df.to_csv(report_dir / "slice_metrics.csv", index=False)
    save_json(report_dir / "summary.json", {"status": "ok", "rows": int(len(merged))})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run slice performance metrics.")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--as_of", default="now")
    args = parser.parse_args()
    run(hours=args.hours, as_of=args.as_of)


if __name__ == "__main__":
    main()
