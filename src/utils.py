from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


FEATURES: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"
SYNTH_FEATURES: List[str] = [
    "amount",
    "currency",
    "country",
    "channel",
    "hour_of_day",
    "is_new_device",
    "is_new_ip",
    "distance_from_home_km",
    "geo_mismatch",
    "user_txn_count_5m",
    "user_txn_count_1h",
    "user_amount_sum_1h",
    "user_avg_amount_30d",
    "merchant_chargeback_rate_30d",
    "device_risk_score",
    "ip_risk_score",
    "drift_phase",
]
SYNTH_TARGET = "is_fraud"
SYNTH_CATEGORICALS = ["currency", "country", "channel"]


def ensure_dir(path: str | Path) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Ensure the Kaggle creditcard.csv format."
        )


def validate_input_payload(payload: Dict[str, Any], feature_list: List[str]) -> None:
    missing = [col for col in feature_list if col not in payload]
    if missing:
        raise ValueError(
            "Input JSON is missing required feature keys: " + ", ".join(missing)
        )


def make_feature_frame(payload: Dict[str, Any], feature_list: List[str]) -> pd.DataFrame:
    validate_input_payload(payload, feature_list)
    row = {feature: payload[feature] for feature in feature_list}
    return pd.DataFrame([row], columns=feature_list)


def get_dataset_config(dataset_type: str) -> Tuple[List[str], str, List[str]]:
    if dataset_type == "kaggle":
        return FEATURES, TARGET, []
    if dataset_type == "synth":
        return SYNTH_FEATURES, SYNTH_TARGET, SYNTH_CATEGORICALS
    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def encode_synth_features(
    df: pd.DataFrame, categorical_cols: List[str], feature_list: List[str] | None = None
) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=categorical_cols, prefix_sep="=")
    if feature_list is None:
        return encoded
    for col in feature_list:
        if col not in encoded.columns:
            encoded[col] = 0
    extra = [col for col in encoded.columns if col not in feature_list]
    if extra:
        encoded = encoded.drop(columns=extra)
    return encoded[feature_list]
