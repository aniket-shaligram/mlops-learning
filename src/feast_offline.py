from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from feast import FeatureStore

from feast_client import FEAST_FEATURE_NAMES
from utils import SYNTH_FEATURES

ENTITY_KEYS = ["user_id", "merchant_id", "device_id", "ip_id"]
NON_FEAST_FEATURES = [feature for feature in SYNTH_FEATURES if feature not in FEAST_FEATURE_NAMES]


def _load_dataset(path: Path, max_rows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    if max_rows:
        df = df.head(max_rows)
    return df


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required + [label] if col not in df.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Ensure the synthetic schema is used."
        )


def _ensure_event_timestamp(df: pd.DataFrame) -> pd.Series:
    if "event_ts" not in df.columns:
        raise ValueError("Dataset is missing required column: event_ts")
    event_ts = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    if event_ts.isna().any():
        raise ValueError("event_ts contains invalid or missing timestamps")
    return event_ts


def build_offline_training_frame(
    data_path: str,
    repo_path: str = "feast_repo",
    max_rows: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    path = Path(data_path)
    df = _load_dataset(path, max_rows)

    event_ts = _ensure_event_timestamp(df)
    _validate_columns(df, ENTITY_KEYS + NON_FEAST_FEATURES, "is_fraud")

    entity_df = df[ENTITY_KEYS].copy()
    entity_df["event_timestamp"] = event_ts

    store = FeatureStore(repo_path=repo_path)
    try:
        hist_df = store.get_historical_features(
            entity_df=entity_df,
            features=["fraud_feature_service"],
        ).to_df()
    except Exception:
        try:
            service = store.get_feature_service("fraud_feature_service")
            feature_refs = [
                f"{proj.name}:{feature.name}"
                for proj in service.feature_view_projections
                for feature in proj.features
            ]
            if not feature_refs:
                raise ValueError("fraud_feature_service has no feature references")
            hist_df = store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs,
            ).to_df()
        except Exception as exc:
            raise ValueError(
                "Feast registry is missing feature service 'fraud_feature_service'. "
                "Run `cd feast_repo && feast apply` and try again."
            ) from exc

    context_df = df[ENTITY_KEYS + NON_FEAST_FEATURES + ["is_fraud"]].copy()
    context_df["event_timestamp"] = event_ts
    context_df = context_df[["event_timestamp"] + ENTITY_KEYS + NON_FEAST_FEATURES + ["is_fraud"]]

    merged = hist_df.merge(
        context_df,
        on=["event_timestamp"] + ENTITY_KEYS,
        how="left",
    )

    missing_features = [
        feature
        for feature in FEAST_FEATURE_NAMES
        if feature not in merged.columns or merged[feature].isna().any()
    ]
    if missing_features:
        raise ValueError(
            "Feast returned missing feature values: " + ", ".join(missing_features)
        )

    X = merged[SYNTH_FEATURES].copy()
    y = merged["is_fraud"].copy()
    return X, y
