from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import psycopg2
from feast import FeatureStore
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))

from feast_client import ENTITY_KEYS, FEAST_FEATURE_REFS
from mlflow_utils import (
    log_artifacts_dir,
    log_dataset_fingerprint,
    log_metrics_dict,
    log_model_generic,
    log_params_dict,
    register_and_promote,
    start_run,
)
from src.models.anomaly_model import AnomalyModel
from src.models.rules_model import RulesModel
from utils import SYNTH_CATEGORICALS, encode_synth_features, ensure_dir, save_json


def _parse_time(value: str) -> datetime:
    raw = value.strip().lower()
    now = datetime.now(timezone.utc)
    if raw == "now":
        return now
    if raw.startswith("now-") or raw.startswith("now+"):
        sign = 1 if raw[3] == "+" else -1
        num = int(raw[4:-1])
        unit = raw[-1]
        delta = {
            "m": timedelta(minutes=num),
            "h": timedelta(hours=num),
            "d": timedelta(days=num),
            "w": timedelta(weeks=num),
        }.get(unit)
        if delta is None:
            raise ValueError(f"Unsupported time unit in {value}")
        return now + (delta * sign)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse datetime: {value}")
    return parsed.to_pydatetime()


def _pg_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "fraud_poc")
    user = os.getenv("POSTGRES_USER", "fraud")
    password = os.getenv("POSTGRES_PASSWORD", "fraud")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _load_base_rows(start: datetime, end: datetime, label_key: str) -> Tuple[pd.DataFrame, str]:
    if label_key not in {"is_fraud", "label"}:
        label_key = "is_fraud"
    use_labels = os.getenv("USE_TXN_LABELS", "false").lower() == "true"
    if use_labels:
        sql = f"""
            select
              v.event_ts,
              v.user_id,
              v.merchant_id,
              v.device_id,
              v.ip_id,
              v.amount,
              v.currency,
              v.country,
              v.channel,
              v.drift_phase,
              coalesce(l.label, (v.payload->>'{label_key}')::int, (v.payload->>'is_fraud')::int, (v.payload->>'label')::int) as label
            from txn_validated v
            left join txn_labels l on v.event_id = l.event_id
            where v.event_ts >= %s and v.event_ts <= %s
            order by v.event_ts asc
        """
    else:
        sql = f"""
            select
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
              coalesce((payload->>'{label_key}')::int, (payload->>'is_fraud')::int, (payload->>'label')::int) as label
            from txn_validated
            where event_ts >= %s and event_ts <= %s
            order by event_ts asc
        """
    with psycopg2.connect(_pg_dsn()) as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (start, end))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        fallback_sql = sql.replace("where event_ts >= %s and event_ts <= %s", "")
        with psycopg2.connect(_pg_dsn()) as conn:
            with conn.cursor() as cursor:
                cursor.execute(fallback_sql)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        if df.empty:
            raise ValueError(
                "No rows returned from txn_validated. Publish events and run GX first."
            )
    label_status = "present"
    if df["label"].isna().all():
        df["label"] = 0
        label_status = "missing"
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["event_ts"])
    return df.reset_index(drop=True), label_status


def _fetch_feast_features(entity_df: pd.DataFrame, repo_path: str) -> pd.DataFrame:
    store = FeatureStore(repo_path=repo_path)
    try:
        return store.get_historical_features(
            entity_df=entity_df,
            features=["fraud_feature_service"],
        ).to_df()
    except Exception:
        return store.get_historical_features(entity_df=entity_df, features=FEAST_FEATURE_REFS).to_df()


def _build_training_frame(
    start: datetime,
    end: datetime,
    label_key: str,
    repo_path: str,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any], pd.Series, str]:
    base_df, label_status = _load_base_rows(start, end, label_key)
    entity_df = base_df[ENTITY_KEYS].copy()
    entity_df["event_timestamp"] = base_df["event_ts"]

    feast_df = _fetch_feast_features(entity_df=entity_df, repo_path=repo_path)
    feast_df = feast_df.rename(columns={"event_timestamp": "event_ts"})
    if "event_ts" in feast_df.columns:
        feast_df["event_ts"] = pd.to_datetime(feast_df["event_ts"], utc=True, errors="coerce")
    base_df["event_ts"] = pd.to_datetime(base_df["event_ts"], utc=True, errors="coerce")

    merged = feast_df.merge(
        base_df,
        on=["event_ts"] + ENTITY_KEYS,
        how="left",
    )

    label = merged["label"].fillna(0).astype(int)
    feature_cols = [
        "amount",
        "currency",
        "country",
        "channel",
        "drift_phase",
    ] + [col for col in feast_df.columns if col not in ["event_ts"] + ENTITY_KEYS]
    feature_cols = [col for col in feature_cols if col in merged.columns]
    X = merged[feature_cols].copy()

    fingerprint = {
        "row_count": int(len(X)),
        "min_event_ts": merged["event_ts"].min().isoformat(),
        "max_event_ts": merged["event_ts"].max().isoformat(),
        "feature_list": feature_cols,
    }
    return X, label, feature_cols, fingerprint, merged["event_ts"], label_status


def _time_split(
    X: pd.DataFrame,
    y: pd.Series,
    timestamps: pd.Series,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) < 50:
        try:
            return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        except ValueError:
            return train_test_split(X, y, test_size=0.2, random_state=seed)
    order = np.argsort(timestamps.values)
    split_idx = int(len(X) * 0.8)
    train_idx = order[:split_idx]
    test_idx = order[split_idx:]
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def _resolve_gbdt(seed: int):
    try:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        )
        return model, "lightgbm"
    except Exception:
        try:
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                eval_metric="logloss",
            )
            return model, "xgboost"
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier

            model = GradientBoostingClassifier(random_state=seed)
            return model, "sklearn_gbdt"


def _evaluate(y_true: pd.Series, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    except Exception:
        metrics["pr_auc"] = 0.0
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    # precision@5%
    k = max(1, int(len(scores) * 0.05))
    topk = np.argsort(scores)[-k:]
    metrics["precision_at_5p"] = float(np.mean(y_true.iloc[topk]))
    # false positive rate at threshold
    negatives = (y_true == 0).sum()
    fp = int(((preds == 1) & (y_true == 0)).sum())
    metrics["fpr"] = float(fp / negatives) if negatives > 0 else 0.0
    return metrics


def _safe_positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.shape[1] == 1:
        return np.zeros(len(X))
    return proba[:, 1]


def _choose_champion(results: Dict[str, Dict[str, float]]) -> str:
    eligible = {k: v for k, v in results.items() if v.get("fpr", 1.0) <= 0.2}
    candidates = eligible or results
    return max(candidates.keys(), key=lambda key: candidates[key].get("pr_auc", 0.0))


def _save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 multi-model training runner.")
    parser.add_argument("--start", default="now-30d")
    parser.add_argument("--end", default="now")
    parser.add_argument("--label_column", default="is_fraud")
    parser.add_argument("--registered_model_name", default="fraud-risk")
    parser.add_argument("--promote_stage", default="Staging")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_dir", default="artifacts_phase4")
    parser.add_argument("--repo_path", default="feast_repo")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.environ.setdefault("POSTGRES_HOST", "localhost")
    os.environ.setdefault("POSTGRES_PORT", "5432")
    os.environ.setdefault("POSTGRES_DB", "fraud_poc")
    os.environ.setdefault("POSTGRES_USER", "fraud")
    os.environ.setdefault("POSTGRES_PASSWORD", "fraud")
    start = _parse_time(args.start)
    end = _parse_time(args.end)
    X_raw, y, feature_cols, fingerprint, event_ts, label_status = _build_training_frame(
        start=start,
        end=end,
        label_key=args.label_column,
        repo_path=args.repo_path,
    )

    timestamps = pd.to_datetime(event_ts, utc=True, errors="coerce")
    X = X_raw
    if any(col in X.columns for col in SYNTH_CATEGORICALS):
        X = encode_synth_features(X, [col for col in SYNTH_CATEGORICALS if col in X.columns])
        feature_cols = list(X.columns)

    X_train, X_test, y_train, y_test = _time_split(X, y, timestamps, args.seed)

    artifacts_root = ensure_dir(args.model_dir)
    run_results: Dict[str, Dict[str, float]] = {}
    model_meta: Dict[str, Dict[str, Any]] = {}

    # Rules model
    rules = RulesModel()
    with start_run("phase4-rules", tags={"model_type": "rules"}):
        scores = rules.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, scores, threshold=0.5)
        log_metrics_dict(metrics)
        log_params_dict({"model_type": "rules", "threshold": 0.5, "label_status": label_status})
        log_dataset_fingerprint(
            row_count=fingerprint["row_count"],
            min_ts=fingerprint["min_event_ts"],
            max_ts=fingerprint["max_event_ts"],
            feature_list=feature_cols,
        )
        rules_dir = artifacts_root / "rules"
        _save_model(rules, rules_dir / "rules_model.pkl")
        save_json(rules_dir / "thresholds.json", rules.thresholds)
        log_artifacts_dir(rules_dir)
        run_results["rules"] = metrics
        model_meta["rules"] = {"model_uri": str(rules_dir / "rules_model.pkl")}

    # Supervised model
    gbdt, gbdt_flavor = _resolve_gbdt(args.seed)
    with start_run("phase4-gbdt", tags={"model_type": gbdt_flavor}):
        if y_train.nunique() < 2:
            gbdt = DummyClassifier(strategy="prior", random_state=args.seed)
            gbdt_flavor = "dummy"
        gbdt.fit(X_train, y_train)
        scores = _safe_positive_scores(gbdt, X_test)
        metrics = _evaluate(y_test, scores, threshold=0.5)
        log_metrics_dict(metrics)
        log_params_dict({"model_type": gbdt_flavor, "label_status": label_status})
        log_dataset_fingerprint(
            row_count=fingerprint["row_count"],
            min_ts=fingerprint["min_event_ts"],
            max_ts=fingerprint["max_event_ts"],
            feature_list=feature_cols,
        )
        gbdt_dir = artifacts_root / "gbdt"
        _save_model(gbdt, gbdt_dir / "model.pkl")
        save_json(gbdt_dir / "features.json", {"features": feature_cols})
        save_json(gbdt_dir / "feature_order.json", {"features": feature_cols})
        log_artifacts_dir(gbdt_dir)
        model_uri = log_model_generic(gbdt, "model", flavor=gbdt_flavor)
        run_results["gbdt"] = metrics
        model_meta["gbdt"] = {"model_uri": model_uri or str(gbdt_dir / "model.pkl")}

    # Anomaly model
    anomaly = AnomalyModel(random_state=args.seed)
    with start_run("phase4-anomaly", tags={"model_type": "isolation_forest"}):
        anomaly.fit(X_train, y_train)
        scores = anomaly.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, scores, threshold=0.5)
        log_metrics_dict(metrics)
        log_params_dict({"model_type": "isolation_forest", "label_status": label_status})
        log_dataset_fingerprint(
            row_count=fingerprint["row_count"],
            min_ts=fingerprint["min_event_ts"],
            max_ts=fingerprint["max_event_ts"],
            feature_list=feature_cols,
        )
        anomaly_dir = artifacts_root / "anomaly"
        _save_model(anomaly, anomaly_dir / "model.pkl")
        save_json(anomaly_dir / "features.json", {"features": feature_cols})
        log_artifacts_dir(anomaly_dir)
        model_uri = log_model_generic(anomaly, "model", flavor="sklearn")
        run_results["anomaly"] = metrics
        model_meta["anomaly"] = {"model_uri": model_uri or str(anomaly_dir / "model.pkl")}

    champion = _choose_champion(run_results)
    champion_info = {
        "model_type": champion,
        "metrics": run_results[champion],
        "model_uri": model_meta[champion]["model_uri"],
        "feature_list": feature_cols,
        "training_window": {"start": start.isoformat(), "end": end.isoformat()},
    }
    save_json(artifacts_root / "champion.json", champion_info)

    registry = register_and_promote(
        model_uri=champion_info["model_uri"],
        registered_model_name=args.registered_model_name,
        stage=args.promote_stage,
        local_artifacts_dir=artifacts_root / champion,
    )
    save_json(artifacts_root / "registry_result.json", registry)

    print(json.dumps({"champion": champion_info, "registry": registry}, indent=2))


if __name__ == "__main__":
    main()
