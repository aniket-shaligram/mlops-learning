from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import pika
import psycopg2
from psycopg2.extras import execute_values

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


def _validate_payload(payload: Dict[str, Any]) -> None:
    required = [
        "event_id",
        "event_type",
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
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume RabbitMQ events into Postgres.")
    parser.add_argument(
        "--amqp_url",
        default="amqp://guest:guest@localhost:5672/%2F",
        help="AMQP connection URL.",
    )
    parser.add_argument(
        "--pg_dsn",
        default="postgresql://fraud:fraud@localhost:5432/fraud_poc",
        help="Postgres DSN.",
    )
    parser.add_argument("--prefetch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    pg_conn = psycopg2.connect(args.pg_dsn)
    pg_conn.autocommit = False

    amqp_conn = pika.BlockingConnection(pika.URLParameters(args.amqp_url))
    channel = amqp_conn.channel()
    _declare_topology(channel)
    channel.basic_qos(prefetch_count=args.prefetch)
    flush_threshold = args.batch_size
    if args.prefetch and args.prefetch < args.batch_size:
        flush_threshold = args.prefetch

    rows: List[Tuple[Any, ...]] = []
    tags: List[int] = []

    def flush() -> None:
        if not rows:
            return
        with pg_conn.cursor() as cursor:
            execute_values(
                cursor,
                """
                insert into txn_raw (
                    event_id,
                    event_type,
                    event_ts,
                    user_id,
                    merchant_id,
                    device_id,
                    ip_id,
                    amount,
                    currency,
                    country,
                    channel,
                    drift_phase,
                    payload
                ) values %s
                on conflict (event_id) do nothing
                """,
                rows,
            )
        pg_conn.commit()
        for tag in tags:
            channel.basic_ack(delivery_tag=tag)
        rows.clear()
        tags.clear()

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        _properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        nonlocal rows, tags
        try:
            payload = json.loads(body.decode("utf-8"))
            _validate_payload(payload)
            row = (
                payload["event_id"],
                payload["event_type"],
                payload["event_ts"],
                int(payload["user_id"]),
                int(payload["merchant_id"]),
                int(payload["device_id"]),
                int(payload["ip_id"]),
                float(payload["amount"]),
                payload["currency"],
                payload["country"],
                payload["channel"],
                int(payload["drift_phase"]),
                json.dumps(payload),
            )
            rows.append(row)
            tags.append(method.delivery_tag)
            if len(rows) >= flush_threshold:
                flush()
        except Exception as exc:
            print(f"ERROR: {exc}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

    print("starting consumer on tx.raw.q")
    try:
        channel.basic_consume(queue=QUEUE, on_message_callback=on_message, auto_ack=False)
        channel.start_consuming()
    except KeyboardInterrupt:
        print("stopping consumer")
    finally:
        try:
            flush()
        except Exception as exc:
            print(f"ERROR flushing batch: {exc}")
            pg_conn.rollback()
        channel.close()
        amqp_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
