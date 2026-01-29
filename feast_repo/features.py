from datetime import timedelta
import os

from feast import Entity, FeatureService, FeatureView, Field

try:
    from feast.data_source import FileSource
except ImportError:  # Feast >= 0.41
    from feast import FileSource
try:
    from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
        PostgreSQLSource,
    )
except Exception:
    try:
        from feast.infra.offline_stores.contrib.postgres_offline_store.postgres import (
            PostgreSQLSource,
        )
    except Exception:  # pragma: no cover - best effort fallback for older Feast
        PostgreSQLSource = None
from feast.types import Float32, Int64
from feast.value_type import ValueType

OFFLINE_SOURCE = os.getenv("FEAST_OFFLINE_SOURCE", "postgres").lower()

if OFFLINE_SOURCE == "file":
    TRANSACTIONS_SOURCE = FileSource(
        path="../data/synth_transactions.parquet",
        timestamp_field="event_ts",
    )
else:
    if PostgreSQLSource is None:
        raise ImportError(
            "PostgreSQLSource is not available. Install a Feast version with "
            "Postgres offline store support."
        )

    TRANSACTIONS_SOURCE = PostgreSQLSource(
        name="txn_validated_source",
        query="""
        select
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
          validated_at,
          count(*) over (
            partition by user_id
            order by event_ts
            range between interval '5 minutes' preceding and current row
          )::bigint as user_txn_count_5m,
          count(*) over (
            partition by user_id
            order by event_ts
            range between interval '1 hour' preceding and current row
          )::bigint as user_txn_count_1h,
          coalesce(
            sum(amount) over (
              partition by user_id
              order by event_ts
              range between interval '1 hour' preceding and current row
            ),
            0.0
          )::double precision as user_amount_sum_1h,
          coalesce(
            avg(amount) over (
              partition by user_id
              order by event_ts
              range between interval '30 days' preceding and current row
            ),
            0.0
          )::double precision as user_avg_amount_30d,
          0.0::double precision as merchant_chargeback_rate_30d,
          0.0::double precision as device_risk_score,
          0.0::double precision as ip_risk_score
        from txn_validated
        """,
        timestamp_field="event_ts",
        created_timestamp_column="validated_at",
    )

user_entity = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.INT64)
merchant_entity = Entity(
    name="merchant_id", join_keys=["merchant_id"], value_type=ValueType.INT64
)
device_entity = Entity(
    name="device_id", join_keys=["device_id"], value_type=ValueType.INT64
)
ip_entity = Entity(name="ip_id", join_keys=["ip_id"], value_type=ValueType.INT64)

user_features = FeatureView(
    name="user_features",
    entities=[user_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="user_txn_count_5m", dtype=Int64),
        Field(name="user_txn_count_1h", dtype=Int64),
        Field(name="user_amount_sum_1h", dtype=Float32),
        Field(name="user_avg_amount_30d", dtype=Float32),
    ],
    online=True,
    source=TRANSACTIONS_SOURCE,
)

merchant_features = FeatureView(
    name="merchant_features",
    entities=[merchant_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="merchant_chargeback_rate_30d", dtype=Float32),
    ],
    online=True,
    source=TRANSACTIONS_SOURCE,
)

device_features = FeatureView(
    name="device_features",
    entities=[device_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="device_risk_score", dtype=Float32),
    ],
    online=True,
    source=TRANSACTIONS_SOURCE,
)

ip_features = FeatureView(
    name="ip_features",
    entities=[ip_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="ip_risk_score", dtype=Float32),
    ],
    online=True,
    source=TRANSACTIONS_SOURCE,
)

fraud_feature_service = FeatureService(
    name="fraud_feature_service",
    features=[user_features, merchant_features, device_features, ip_features],
)
