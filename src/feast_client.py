from __future__ import annotations

from typing import Dict, List

from feast import FeatureStore

ENTITY_KEYS = ["user_id", "merchant_id", "device_id", "ip_id"]
FEAST_FEATURE_REFS = [
    "user_features:user_txn_count_5m",
    "user_features:user_txn_count_1h",
    "user_features:user_amount_sum_1h",
    "user_features:user_avg_amount_30d",
    "merchant_features:merchant_chargeback_rate_30d",
    "device_features:device_risk_score",
    "ip_features:ip_risk_score",
]
FEAST_FEATURE_NAMES = [ref.split(":")[1] for ref in FEAST_FEATURE_REFS]


def _require_entities(payload: Dict[str, object], keys: List[str]) -> Dict[str, object]:
    missing = [key for key in keys if key not in payload]
    if missing:
        raise ValueError(
            "Input payload is missing required Feast entity keys: "
            + ", ".join(missing)
        )
    return {key: payload[key] for key in keys}


def get_online_feature_vector(
    payload: Dict[str, object], repo_path: str = "feast_repo"
) -> Dict[str, object]:
    entity_row = _require_entities(payload, ENTITY_KEYS)
    store = FeatureStore(repo_path=repo_path)
    response = store.get_online_features(
        features=FEAST_FEATURE_REFS,
        entity_rows=[entity_row],
    ).to_dict()

    features: Dict[str, object] = {}
    missing_features = []
    for name in FEAST_FEATURE_NAMES:
        value = response.get(name)
        if isinstance(value, list):
            value = value[0] if value else None
        if value is None:
            missing_features.append(name)
        else:
            features[name] = value

    if missing_features:
        raise ValueError(
            "Feast returned missing feature values: " + ", ".join(missing_features)
        )
    return features
