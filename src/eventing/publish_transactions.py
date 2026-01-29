from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import pandas as pd
import pika

EXCHANGE = "tx.events"
ROUTING_KEY = "txn.created"
QUEUE = "tx.raw.q"
DLX = "tx.dlx"
DLQ = "tx.raw.dlq"


def _declare_topology(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=DLX, exchange_type="fanout", durable=True)
    channel.queue_declare(
        queue=QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX},
    )
    channel.queue_declare(queue=DLQ, durable=True)
    channel.queue_bind(queue=QUEUE, exchange=EXCHANGE, routing_key=ROUTING_KEY)
    channel.queue_bind(queue=DLQ, exchange=DLX)


def _connect_with_retry(amqp_url: str, attempts: int = 10) -> pika.BlockingConnection:
    delay = 0.5
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return pika.BlockingConnection(pika.URLParameters(amqp_url))
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 8)
    raise RuntimeError("Unable to connect to RabbitMQ") from last_exc


def _load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _to_iso_utc(value: object) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid event_ts value: {value}")
    return ts.isoformat().replace("+00:00", "Z")


def _derive_label(row: pd.Series) -> int:
    if "is_fraud" in row and pd.notna(row["is_fraud"]):
        return int(row["is_fraud"])
    score = 0.0
    amount = float(row.get("amount", 0.0) or 0.0)
    if amount >= 500:
        score += 1.0
    if int(row.get("geo_mismatch", 0) or 0) == 1:
        score += 1.0
    if int(row.get("is_new_device", 0) or 0) == 1:
        score += 0.75
    if int(row.get("is_new_ip", 0) or 0) == 1:
        score += 0.75
    if str(row.get("channel", "") or "").lower() == "mobile":
        score += 0.25
    return int(score >= 1.0)


def _iter_rows(df: pd.DataFrame, max_events: int) -> Iterable[pd.Series]:
    if max_events > 0:
        df = df.head(max_events)
    for _, row in df.iterrows():
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish synthetic transactions to RabbitMQ.")
    parser.add_argument("--data_path", required=True, help="Path to parquet/csv dataset.")
    parser.add_argument(
        "--amqp_url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="AMQP connection URL.",
    )
    parser.add_argument("--rate", type=int, default=5000, help="Events per second.")
    parser.add_argument("--max_events", type=int, default=0, help="Max events to send (0=all).")
    args = parser.parse_args()

    df = _load_dataframe(Path(args.data_path))
    required = [
        "event_ts",
        "user_id",
        "merchant_id",
        "device_id",
        "ip_id",
        "amount",
        "currency",
        "country",
        "channel",
        "drift_phase",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    connection = _connect_with_retry(args.amqp_url)
    channel = connection.channel()
    _declare_topology(channel)
    channel.confirm_delivery()

    interval = 1.0 / args.rate if args.rate > 0 else 0.0
    sent = 0
    last_time = time.monotonic()

    for row in _iter_rows(df, args.max_events):
        payload = {
            "event_id": str(uuid4()),
            "event_type": "txn.created",
            "event_ts": _to_iso_utc(row["event_ts"]),
            "user_id": int(row["user_id"]),
            "merchant_id": int(row["merchant_id"]),
            "device_id": int(row["device_id"]),
            "ip_id": int(row["ip_id"]),
            "amount": float(row["amount"]),
            "currency": str(row["currency"]),
            "country": str(row["country"]),
            "channel": str(row["channel"]),
            "drift_phase": int(row["drift_phase"]),
            "is_fraud": _derive_label(row),
        }
        body = json.dumps(payload).encode("utf-8")
        ok = channel.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY,
            body=body,
            mandatory=True,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        if ok is False:
            raise RuntimeError("Publish not confirmed by broker")
        if ok is None:
            waiter = getattr(channel, "wait_for_confirms", None)
            if waiter is not None and not waiter():
                raise RuntimeError("Publish not confirmed by broker")

        sent += 1
        if sent % 10_000 == 0:
            print(f"published {sent} events")

        if interval > 0:
            elapsed = time.monotonic() - last_time
            sleep_for = interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_time = time.monotonic()

    print(f"done publishing {sent} events")
    connection.close()


if __name__ == "__main__":
    main()
