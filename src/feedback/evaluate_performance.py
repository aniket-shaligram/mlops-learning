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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.utils import ensure_dir, save_json


def _pg_dsn() -> str:
    return os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")


def _parse_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _window(as_of: datetime, hours: int) -> Tuple[datetime, datetime]:
    end = as_of
    start = as_of - timedelta(hours=hours)
    return start, end


def _fetch_rows(start: datetime, end: datetime) -> pd.DataFrame:
    sql = """
        select final_score, decision, country, channel, drift_phase, served_by, true_label
        from decision_with_labels
        where labeled_at >= %s and labeled_at < %s
          and true_label is not null
    """
    with psycopg2.connect(_pg_dsn()) as conn:
        return pd.read_sql(sql, conn, params=(start, end))


def _confusion(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def evaluate(
    window_hours: int = 24,
    as_of: str = "now",
    threshold: float = 0.7,
) -> Dict[str, Any]:
    as_of_dt = _parse_time(as_of)
    start, end = _window(as_of_dt, window_hours)
    df = _fetch_rows(start, end)

    if df.empty or df["true_label"].nunique() < 2 or len(df) < 200:
        return {
            "status": "insufficient_data",
            "rows": int(len(df)),
            "perf_ok": None,
            "reason": "insufficient labeled data",
        }

    y_true = df["true_label"].astype(int)
    y_pred = (df["final_score"] >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    auc = None
    try:
        auc = float(roc_auc_score(y_true, df["final_score"]))
    except Exception:
        auc = None

    overall = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "threshold": float(threshold),
        "rows": int(len(df)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        **_confusion(y_true, y_pred),
    }

    slices = []
    for col in ["country", "channel", "drift_phase", "served_by"]:
        for value, group in df.groupby(col):
            y_t = group["true_label"].astype(int)
            y_p = (group["final_score"] >= threshold).astype(int)
            p, r, f, _ = precision_recall_fscore_support(
                y_t, y_p, average="binary", zero_division=0
            )
            slices.append(
                {
                    "slice": col,
                    "value": value,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f),
                    "count": int(len(group)),
                }
            )

    return {
        "status": "ok",
        "overall": overall,
        "slices": slices,
        "perf_ok": True,
        "reason": "ok",
    }


def run(window_hours: int = 24, as_of: str = "now", threshold: float = 0.7) -> Dict[str, Any]:
    as_of_dt = _parse_time(as_of)
    report_dir = ensure_dir(
        Path("feedback") / "reports" / "perf" / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )
    results = evaluate(window_hours=window_hours, as_of=as_of, threshold=threshold)

    if results["status"] != "ok":
        health = {
            "perf_ok": results.get("perf_ok"),
            "reason": results.get("reason"),
            "rows": results.get("rows", 0),
        }
        save_json(Path(report_dir) / "overall.json", results)
        save_json(Path(report_dir) / "health.json", health)
        return results

    overall = results["overall"]
    pd.DataFrame(results["slices"]).to_csv(Path(report_dir) / "slices.csv", index=False)
    save_json(Path(report_dir) / "overall.json", overall)
    save_json(Path(report_dir) / "health.json", {"perf_ok": True, "reason": "ok"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rolling performance.")
    parser.add_argument("--window_hours", type=int, default=24)
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    results = run(
        window_hours=args.window_hours,
        as_of=args.as_of,
        threshold=args.threshold,
    )
    print(json.dumps(results if results.get("status") != "ok" else results.get("overall", {}), indent=2))


if __name__ == "__main__":
    main()
