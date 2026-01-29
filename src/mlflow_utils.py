from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from contextlib import nullcontext


def _safe_import_mlflow():
    try:
        import mlflow
    except Exception:
        return None
    return mlflow


def start_run(run_name: str, tags: Optional[Dict[str, str]] = None):
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return nullcontext()
    run = mlflow.start_run(run_name=run_name)
    if tags:
        mlflow.set_tags(tags)
    return run


def log_dataset_fingerprint(
    row_count: int,
    min_ts: Optional[str],
    max_ts: Optional[str],
    feature_list: Iterable[str],
) -> Dict[str, Any]:
    payload = {
        "row_count": int(row_count),
        "min_event_ts": min_ts,
        "max_event_ts": max_ts,
        "feature_list": list(feature_list),
    }
    feature_hash = hashlib.sha256(json.dumps(payload["feature_list"]).encode("utf-8")).hexdigest()
    payload["feature_list_sha256"] = feature_hash
    mlflow = _safe_import_mlflow()
    if mlflow is not None:
        mlflow.log_dict(payload, "dataset_fingerprint.json")
        mlflow.log_params(
            {
                "row_count": int(row_count),
                "min_event_ts": min_ts or "",
                "max_event_ts": max_ts or "",
                "feature_list_sha256": feature_hash,
            }
        )
    return payload


def log_metrics_dict(metrics: Dict[str, float]) -> None:
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))


def log_params_dict(params: Dict[str, Any]) -> None:
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_artifacts_dir(path: str | Path) -> None:
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return
    mlflow.log_artifacts(str(path))


def log_model_generic(model: Any, model_name: str, flavor: str = "sklearn") -> Optional[str]:
    mlflow = _safe_import_mlflow()
    if mlflow is None:
        return None
    model_uri = None
    try:
        if flavor == "lightgbm":
            import mlflow.lightgbm

            mlflow.lightgbm.log_model(model, model_name)
        elif flavor == "xgboost":
            import mlflow.xgboost

            mlflow.xgboost.log_model(model, model_name)
        else:
            import mlflow.sklearn

            mlflow.sklearn.log_model(model, model_name)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
    except Exception:
        model_uri = None
    return model_uri


def register_and_promote(
    model_uri: str,
    registered_model_name: str,
    stage: str,
    local_artifacts_dir: Optional[str | Path] = None,
) -> Dict[str, str]:
    mlflow = _safe_import_mlflow()
    if mlflow is not None:
        try:
            client = mlflow.tracking.MlflowClient()
            result = mlflow.register_model(model_uri, registered_model_name)
            client.transition_model_version_stage(
                name=registered_model_name,
                version=result.version,
                stage=stage,
                archive_existing_versions=True,
            )
            return {
                "mode": "mlflow_registry",
                "model_uri": model_uri,
                "registered_model": registered_model_name,
                "stage": stage,
                "version": str(result.version),
            }
        except Exception:
            pass

    registry_root = Path("registry") / registered_model_name / "champion"
    registry_root.mkdir(parents=True, exist_ok=True)
    if local_artifacts_dir:
        src_path = Path(local_artifacts_dir)
        if src_path.exists():
            for item in src_path.iterdir():
                dest = registry_root / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
    pointer = {
        "mode": "local_registry",
        "model_uri": model_uri,
        "registered_model": registered_model_name,
        "stage": stage,
        "path": str(registry_root),
    }
    pointer_path = registry_root / "metadata.json"
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True), encoding="utf-8")
    return pointer
