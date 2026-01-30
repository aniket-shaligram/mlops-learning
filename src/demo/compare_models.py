from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import psycopg2


def _pg_dsn() -> str:
    return os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")


def _has_column(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "select 1 from information_schema.columns where table_name=%s and column_name=%s",
            (table, column),
        )
        return cur.fetchone() is not None


def _window(minutes: int) -> Tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    return start, end


def _summary(conn, start: datetime, end: datetime, decision_filter: str | None) -> List[Dict[str, object]]:
    served_by = _has_column(conn, "decision_log", "served_by")
    group_col = "served_by" if served_by else "model_versions->>'champion_type'"
    where_clause = "created_at >= %s and created_at < %s"
    params = [start, end]
    if decision_filter:
        where_clause += " and decision = %s"
        params.append(decision_filter)
    sql = f"""
        select {group_col} as model_key,
               count(*) as cnt,
               avg(final_score) as avg_score,
               avg(case when decision='block' then 1 else 0 end) as block_rate,
               avg(case when decision='review' then 1 else 0 end) as review_rate,
               avg(case when decision='step_up' then 1 else 0 end) as step_up_rate
        from decision_log
        where {where_clause}
        group by model_key
        order by cnt desc
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [
            {
                "model": row[0] or "unknown",
                "count": int(row[1]),
                "avg_score": float(row[2]) if row[2] is not None else None,
                "block_rate": float(row[3]) if row[3] is not None else None,
                "review_rate": float(row[4]) if row[4] is not None else None,
                "step_up_rate": float(row[5]) if row[5] is not None else None,
            }
            for row in rows
        ]


def _precision_recall(conn, start: datetime, end: datetime, decision_filter: str | None) -> Dict[str, Dict[str, float]]:
    served_by = _has_column(conn, "decision_log", "served_by")
    group_col = "served_by" if served_by else "model_versions->>'champion_type'"
    where_clause = "d.created_at >= %s and d.created_at < %s"
    params = [start, end]
    if decision_filter:
        where_clause += " and d.decision = %s"
        params.append(decision_filter)
    sql = f"""
        select {group_col} as model_key, d.final_score, l.label
        from decision_log d
        join txn_labels l on d.event_id = l.event_id
        where {where_clause}
    """
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    by_model: Dict[str, List[Tuple[float, int]]] = {}
    for model_key, score, label in rows:
        by_model.setdefault(model_key or "unknown", []).append((float(score), int(label)))
    results = {}
    for model_key, items in by_model.items():
        tp = fp = fn = 0
        for score, label in items:
            pred = int(score >= 0.7)
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        results[model_key] = {"precision": precision, "recall": recall, "count": len(items)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recent model performance.")
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--decision", help="Filter by decision (e.g., approve/review)")
    args = parser.parse_args()

    start, end = _window(args.minutes)
    with psycopg2.connect(_pg_dsn()) as conn:
        summary = _summary(conn, start, end, args.decision)
        labels = _precision_recall(conn, start, end, args.decision)

    output = {"window_minutes": args.minutes, "summary": summary}
    if labels:
        output["labels"] = labels
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
