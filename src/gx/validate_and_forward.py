from __future__ import annotations

import argparse
import json
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import great_expectations as ge
import pandas as pd
import pika
import psycopg2
from psycopg2.extras import Json, execute_values

EXCHANGE = "tx.events"
ROUTING_KEY = "txn.created"
QUEUE = "tx.raw.q"

VALIDATED_EXCHANGE = "tx.validated"
VALIDATED_ROUTING_KEY = "txn.validated"
VALIDATED_QUEUE = "tx.validated.q"

QUARANTINE_EXCHANGE = "tx.quarantine"
QUARANTINE_ROUTING_KEY = "txn.quarantined"
QUARANTINE_QUEUE = "tx.quarantine.q"

DLX = "tx.dlx"
DLQ = "tx.raw.dlq"

SUITE_NAME = "txn_event_suite"
REQUIRED_COLUMNS = [
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

VOLUME_WINDOW_SECONDS = 60
VOLUME_SPIKE_THRESHOLD = 20000


def _declare_topology(channel: pika.adapters.blocking_connection.BlockingChannel) -> None:
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=VALIDATED_EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=QUARANTINE_EXCHANGE, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=DLX, exchange_type="fanout", durable=True)

    channel.queue_declare(
        queue=QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX},
    )
    channel.queue_declare(
        queue=VALIDATED_QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX},
    )
    channel.queue_declare(
        queue=QUARANTINE_QUEUE,
        durable=True,
        arguments={"x-dead-letter-exchange": DLX},
    )
    channel.queue_declare(queue=DLQ, durable=True)

    channel.queue_bind(queue=QUEUE, exchange=EXCHANGE, routing_key=ROUTING_KEY)
    channel.queue_bind(queue=VALIDATED_QUEUE, exchange=VALIDATED_EXCHANGE, routing_key=VALIDATED_ROUTING_KEY)
    channel.queue_bind(queue=QUARANTINE_QUEUE, exchange=QUARANTINE_EXCHANGE, routing_key=QUARANTINE_ROUTING_KEY)
    channel.queue_bind(queue=DLQ, exchange=DLX)


def _connect_rabbit_with_retry(amqp_url: str, attempts: int = 10) -> pika.BlockingConnection:
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


def _connect_postgres_with_retry(pg_dsn: str, attempts: int = 10) -> psycopg2.extensions.connection:
    delay = 0.5
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return psycopg2.connect(pg_dsn)
        except Exception as exc:
            last_exc = exc
            if attempt == attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 8)
    raise RuntimeError("Unable to connect to Postgres") from last_exc


def _publish_or_raise(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    exchange: str,
    routing_key: str,
    payload: Dict[str, Any],
) -> None:
    body = json.dumps(payload).encode("utf-8")
    ok = channel.basic_publish(
        exchange=exchange,
        routing_key=routing_key,
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


def _summarize_failures(results: Dict[str, Any]) -> str:
    failures: List[str] = []
    for item in results.get("results", []):
        if not item.get("success", True):
            expectation = item.get("expectation_config", {}).get("expectation_type", "unknown")
            failures.append(expectation)
    if not failures:
        return "validation_failed"
    return "validation_failed: " + ", ".join(sorted(set(failures)))


def _make_quarantine_id(payload: Dict[str, Any]) -> str:
    event_id = payload.get("event_id")
    if event_id:
        return str(event_id)
    return f"unknown:{int(time.time() * 1000)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate events with GX and forward.")
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
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--flush_interval_sec", type=float, default=1.0)
    args = parser.parse_args()

    pg_conn = _connect_postgres_with_retry(args.pg_dsn)
    pg_conn.autocommit = False

    amqp_conn = _connect_rabbit_with_retry(args.amqp_url)
    channel = amqp_conn.channel()
    _declare_topology(channel)
    channel.confirm_delivery()
    channel.basic_qos(prefetch_count=args.prefetch)

    context = ge.get_context(context_root_dir="gx")
    recent_events: deque[float] = deque()
    validated_count = 0
    quarantined_count = 0
    error_count = 0
    processed = 0
    last_flush = time.monotonic()
    validated_rows: List[Tuple[Any, ...]] = []
    validated_tags: List[int] = []
    quarantined_rows: List[Tuple[Any, ...]] = []
    quarantined_tags: List[int] = []

    def _record_volume() -> bool:
        now = time.time()
        recent_events.append(now)
        cutoff = now - VOLUME_WINDOW_SECONDS
        while recent_events and recent_events[0] < cutoff:
            recent_events.popleft()
        return len(recent_events) > VOLUME_SPIKE_THRESHOLD

    def _flush_buffers() -> None:
        nonlocal last_flush
        if not validated_rows and not quarantined_rows:
            return
        try:
            with pg_conn.cursor() as cursor:
                if validated_rows:
                    execute_values(
                        cursor,
                        """
                        insert into txn_validated (
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
                        validated_rows,
                    )
                if quarantined_rows:
                    execute_values(
                        cursor,
                        """
                        insert into txn_quarantine (
                            event_id,
                            reason,
                            payload
                        ) values %s
                        on conflict (event_id) do nothing
                        """,
                        quarantined_rows,
                    )
            pg_conn.commit()
            for tag in validated_tags + quarantined_tags:
                channel.basic_ack(delivery_tag=tag)
        except Exception as exc:
            pg_conn.rollback()
            for tag in validated_tags + quarantined_tags:
                channel.basic_nack(delivery_tag=tag, requeue=False)
            print(f"ERROR flushing batch: {exc}")
        finally:
            validated_rows.clear()
            validated_tags.clear()
            quarantined_rows.clear()
            quarantined_tags.clear()
            last_flush = time.monotonic()

    def on_message(
        ch: pika.adapters.blocking_connection.BlockingChannel,
        method: pika.spec.Basic.Deliver,
        _properties: pika.spec.BasicProperties,
        body: bytes,
    ) -> None:
        nonlocal validated_count, quarantined_count, error_count, processed
        try:
            payload = json.loads(body.decode("utf-8"))
            df = pd.DataFrame([payload])
            df = df.reindex(columns=REQUIRED_COLUMNS)
            validator = context.get_validator(
                batch_data=df,
                expectation_suite_name=SUITE_NAME,
            )
            results = validator.validate().to_json_dict()
            volume_spike = _record_volume()

            if results.get("success") and not volume_spike:
                _publish_or_raise(channel, VALIDATED_EXCHANGE, VALIDATED_ROUTING_KEY, payload)
                validated_rows.append(
                    (
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
                        Json(payload),
                    )
                )
                validated_tags.append(method.delivery_tag)
                validated_count += 1
            else:
                reason = _summarize_failures(results)
                if volume_spike:
                    reason = reason + "; volume_spike"
                quarantine_event_id = _make_quarantine_id(payload)
                quarantine_payload = {
                    "event_id": quarantine_event_id,
                    "reason": reason,
                    "payload": payload,
                }
                _publish_or_raise(channel, QUARANTINE_EXCHANGE, QUARANTINE_ROUTING_KEY, quarantine_payload)
                quarantined_rows.append((quarantine_event_id, reason, Json(payload)))
                quarantined_tags.append(method.delivery_tag)
                quarantined_count += 1
        except Exception as exc:
            error_count += 1
            print(f"ERROR: {exc}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        finally:
            processed += 1
            if (
                len(validated_rows) + len(quarantined_rows) >= args.batch_size
                or (time.monotonic() - last_flush) >= args.flush_interval_sec
            ):
                _flush_buffers()
            if args.log_every and processed % args.log_every == 0:
                print(
                    "counts - validated: "
                    f"{validated_count}, quarantined: {quarantined_count}, errors: {error_count}"
                )

    print("starting GX validator on tx.raw.q")
    try:
        channel.basic_consume(queue=QUEUE, on_message_callback=on_message, auto_ack=False)
        channel.start_consuming()
    except KeyboardInterrupt:
        print("stopping validator")
    finally:
        _flush_buffers()
        channel.close()
        amqp_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
