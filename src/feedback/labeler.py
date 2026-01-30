from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import psycopg2
from psycopg2.extras import execute_values

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


def _payload_label(payload: Any) -> Tuple[int | None, str | None]:
    if payload is None:
        return None, None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            return None, None
    if isinstance(payload, dict):
        for key in ("is_fraud", "label", "isFraud"):
            if key in payload and payload[key] is not None:
                return int(payload[key]), "synthetic_ground_truth"
    return None, None


def _proxy_label(event_id: str, drift_phase: int, amount: float, channel: str) -> int:
    h = int(hashlib.sha256(event_id.encode("utf-8")).hexdigest(), 16)
    jitter = (h % 100) / 1000.0
    score = 0.0
    if drift_phase == 1:
        score += 0.5
    if amount >= 1000:
        score += 0.4
    if channel.lower() == "mobile":
        score += 0.2
    score += jitter
    return int(score >= 0.75)


def run(
    delay_minutes: int = 5,
    batch_size: int = 500,
    as_of: str = "now",
) -> Dict[str, Any]:
    as_of_dt = _parse_time(as_of)
    cutoff = as_of_dt - timedelta(minutes=delay_minutes)

    sql = """
        select d.event_id, d.created_at, d.amount, d.channel, d.drift_phase, v.payload
        from decision_log d
        left join txn_validated v on d.event_id = v.event_id
        where d.event_id is not null
          and d.created_at <= %s
          and not exists (select 1 from txn_labels l where l.event_id = d.event_id)
        order by d.created_at asc
        limit %s
    """
    rows = []
    with psycopg2.connect(_pg_dsn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (cutoff, batch_size))
            rows = cursor.fetchall()

    inserts: list[tuple] = []
    for event_id, _created_at, amount, channel, drift_phase, payload in rows:
        label, source = _payload_label(payload)
        if label is None:
            label = _proxy_label(str(event_id), int(drift_phase or 0), float(amount or 0.0), str(channel or ""))
            source = "proxy_rule"
        inserts.append((event_id, label, source))

    inserted = 0
    if inserts:
        insert_sql = """
            insert into txn_labels (event_id, label, label_source)
            values %s
            on conflict (event_id) do nothing
        """
        with psycopg2.connect(_pg_dsn()) as conn:
            with conn.cursor() as cursor:
                execute_values(cursor, insert_sql, inserts)
                inserted = cursor.rowcount

    report_dir = ensure_dir(
        Path("feedback") / "reports" / "labeler" / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )
    summary = {
        "cutoff": cutoff.isoformat(),
        "requested": int(batch_size),
        "found": int(len(rows)),
        "inserted": int(inserted),
    }
    save_json(Path(report_dir) / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Label scored events after a delay.")
    parser.add_argument("--delay_minutes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--as_of", default="now")
    args = parser.parse_args()
    summary = run(
        delay_minutes=args.delay_minutes,
        batch_size=args.batch_size,
        as_of=args.as_of,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
