from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import (
    ensure_dir,
    encode_synth_features,
    get_dataset_config,
    load_dataset,
    save_json,
    validate_columns,
)


def _resolve_model(
    model_type: str, scale_pos_weight: float, random_seed: int
) -> Tuple[Any, str, str]:
    if model_type in {"lightgbm", "auto"}:
        try:
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            )
            return model, "lightgbm", "scale_pos_weight"
        except Exception as exc:
            if model_type == "lightgbm":
                raise ImportError("LightGBM is not available.") from exc

    if model_type in {"xgboost", "auto"}:
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_seed,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
            )
            return model, "xgboost", "scale_pos_weight"
        except Exception as exc:
            if model_type == "xgboost":
                raise ImportError("XGBoost is not available.") from exc

    if model_type in {"logreg", "auto"}:
        try:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=random_seed,
            )
            return model, "logreg", "class_weight"
        except Exception as exc:
            raise ImportError("scikit-learn LogisticRegression is not available.") from exc

    raise ValueError(f"Unsupported model_type: {model_type}")


def _log_mlflow(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts_dir: Path,
    model: Any,
) -> None:
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:
        return

    with mlflow.start_run():
        for key, value in params.items():
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_artifacts(str(artifacts_dir))
        mlflow.sklearn.log_model(model, "model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a fraud scoring model.")
    parser.add_argument("--data_path", required=True, help="Path to creditcard.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--model_type",
        default="auto",
        choices=["auto", "lightgbm", "xgboost", "logreg"],
    )
    parser.add_argument(
        "--dataset_type",
        default="kaggle",
        choices=["kaggle", "synth"],
        help="Dataset schema to use.",
    )
    parser.add_argument(
        "--use_feast_offline",
        choices=["true", "false"],
        default="false",
        help="Use Feast offline features for synth training.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Limit training rows (synth + Feast offline only).",
    )
    parser.add_argument(
        "--dev_100k",
        action="store_true",
        help="Use 100k rows for Feast offline dev runs.",
    )
    parser.add_argument("--artifacts_dir", default="artifacts")
    args = parser.parse_args()

    use_feast_offline = args.use_feast_offline.lower() == "true"
    if args.dataset_type != "synth" and use_feast_offline:
        raise ValueError("Feast offline features are only supported for synth.")

    artifacts_dir = ensure_dir(args.artifacts_dir)
    feature_list, target_col, categorical_cols = get_dataset_config(args.dataset_type)
    max_rows = args.max_rows
    if max_rows is None and args.dev_100k:
        max_rows = 100000

    if args.dataset_type == "synth" and use_feast_offline:
        from feast_offline import build_offline_training_frame

        X_raw, y = build_offline_training_frame(args.data_path, max_rows=max_rows)
    else:
        df = load_dataset(args.data_path)
        if args.dataset_type == "kaggle":
            validate_columns(df)
        else:
            missing = [col for col in feature_list + [target_col] if col not in df.columns]
            if missing:
                raise ValueError(
                    "Dataset is missing required columns: "
                    + ", ".join(missing)
                    + ". Ensure the synthetic schema is used."
                )

        X_raw = df[feature_list]
        y = df[target_col]
    if args.dataset_type == "synth" and categorical_cols:
        X = encode_synth_features(X_raw, categorical_cols)
    else:
        X = X_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=y,
    )

    positives = int(y_train.sum())
    negatives = int((y_train == 0).sum())
    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0

    model, resolved_type, imbalance_strategy = _resolve_model(
        args.model_type, scale_pos_weight, args.random_seed
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, proba)
    if thresholds.size == 0:
        optimal_threshold = 0.5
    else:
        f1_scores = (2 * precision_curve * recall_curve) / (
            precision_curve + recall_curve + 1e-12
        )
        best_index = int(np.argmax(f1_scores[:-1]))
        optimal_threshold = float(thresholds[best_index])

    preds = (proba >= optimal_threshold).astype(int)

    pr_auc = average_precision_score(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="binary", zero_division=0
    )

    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "optimal_threshold": float(optimal_threshold),
    }

    metadata = {
        "model_type": resolved_type,
        "dataset_type": args.dataset_type,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "imbalance_strategy": imbalance_strategy,
        "scale_pos_weight": float(scale_pos_weight),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_fraud_count": positives,
        "train_legit_count": negatives,
    }
    if categorical_cols:
        metadata["categorical_cols"] = categorical_cols

    model_path = artifacts_dir / "model.pkl"
    joblib.dump(model, model_path)
    save_json(artifacts_dir / "features.json", {"features": list(X.columns)})
    save_json(artifacts_dir / "metrics.json", {**metrics, **metadata})
    save_json(artifacts_dir / "run_config.json", metadata)

    params = {
        "model_type": resolved_type,
        "dataset_type": args.dataset_type,
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "imbalance_strategy": imbalance_strategy,
    }
    _log_mlflow(params, metrics, artifacts_dir, model)

    print(json.dumps({"artifacts_dir": str(artifacts_dir)}, indent=2))


if __name__ == "__main__":
    main()
