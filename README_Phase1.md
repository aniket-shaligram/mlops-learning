# Phase 1 — Data generation + eventing (RabbitMQ → Postgres)

This phase wires a local event pipeline:
synthetic generator → RabbitMQ → consumer → Postgres `txn_raw`.

## 1) Start infra
```bash
docker compose -f ops/docker-compose.yml up -d
```

RabbitMQ UI: `http://localhost:15672` (guest/guest).
If Redis is already running on your machine, stop it or remove the container using port `6379`.

## 2) Create raw landing table
If you don't have `psql` locally, run it inside the container:
```bash
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -f /ops/sql/001_create_raw.sql
```

## 3) Run the consumer
```bash
python src/eventing/consume_to_postgres.py
```

## 4) Generate synthetic data
```bash
python src/synth/generate_synth.py --rows 100000 --format parquet
```

## 5) Publish events
```bash
python src/eventing/publish_transactions.py \
  --data_path data/synth_transactions.parquet \
  --rate 5000 \
  --max_events 0
```

## 6) Verify counts
If you don't have `psql` locally, run it inside the container:
```bash
docker exec -i $(docker ps -qf "name=postgres") psql -U fraud -d fraud_poc -c "select count(*) from txn_raw;"
```

## Notes
- Messages are durable and published to exchange `tx.events` with routing key `txn.created`.
- The consumer inserts with `on conflict do nothing` on `event_id`.
- Invalid messages are nacked and routed to `tx.raw.dlq`.
