from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from src.models.anomaly_model import AnomalyModel
from src.models.rules_model import RulesModel


@dataclass
class ModelBundle:
    rules_model: RulesModel
    champion_model: Optional[Any]
    anomaly_model: Optional[Any]
    metadata: Dict[str, Any]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_joblib(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except ModuleNotFoundError:
        return None


def _resolve_local_champion_path(
    model_type: str,
    registered_model_name: Optional[str],
) -> Optional[Path]:
    if registered_model_name:
        registry_path = Path("registry") / registered_model_name / "champion" / "model.pkl"
        if registry_path.exists():
            return registry_path
    artifacts_path = Path("artifacts_phase4") / model_type / "model.pkl"
    if artifacts_path.exists():
        return artifacts_path
    return None


def _load_mlflow_model(model_uri: str) -> Optional[Any]:
    try:
        import mlflow

        return mlflow.pyfunc.load_model(model_uri)
    except Exception:
        return None


def load_models() -> ModelBundle:
    rules_model = RulesModel()

    anomaly_path = Path("artifacts_phase4") / "anomaly" / "model.pkl"
    anomaly_model = _load_joblib(anomaly_path)
    if anomaly_model is None:
        anomaly_model = AnomalyModel()

    champion_info = _read_json(Path("artifacts_phase4") / "champion.json") or {}
    registry_info = _read_json(Path("artifacts_phase4") / "registry_result.json") or {}

    champion_type = champion_info.get("model_type")
    champion_uri = champion_info.get("model_uri")
    registered_model_name = registry_info.get("registered_model")
    registry_mode = registry_info.get("mode", "local_registry")

    champion_model = None
    if champion_uri:
        champion_model = _load_mlflow_model(champion_uri)
    if champion_model is None and champion_type:
        local_path = _resolve_local_champion_path(champion_type, registered_model_name)
        champion_model = _load_joblib(local_path) if local_path else None

    metadata = {
        "champion_type": champion_type,
        "champion_uri": champion_uri,
        "registry_mode": registry_mode,
        "registered_model_name": registered_model_name,
        "feature_list": champion_info.get("feature_list", []),
    }
    return ModelBundle(
        rules_model=rules_model,
        champion_model=champion_model,
        anomaly_model=anomaly_model,
        metadata=metadata,
    )
