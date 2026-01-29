from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


FEATURES: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"


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
