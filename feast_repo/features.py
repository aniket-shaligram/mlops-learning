from datetime import timedelta


from feast import Entity, FeatureService, FeatureView, Field

try:
    from feast.data_source import FileSource
except ImportError:  # Feast >= 0.41
    from feast import FileSource
from feast.types import Float32, Int64
from feast.value_type import ValueType

TRANSACTIONS_SOURCE = FileSource(
    path="../data/synth_transactions.parquet",
    timestamp_field="event_ts",
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
