from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import psycopg2


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _pg_dsn() -> str:
    return os.getenv("PG_DSN", "postgresql://fraud:fraud@localhost:5432/fraud_poc")


def _fetch_db(transaction_id: str | None) -> dict:
    with psycopg2.connect(_pg_dsn()) as conn:
        with conn.cursor() as cursor:
            if transaction_id:
                cursor.execute(
                    """
                    select decision_id, event_id, event_ts, final_score, decision, served_by, model_versions, scores, features, created_at
                    from decision_log
                    where event_id = %s
                    order by created_at desc
                    limit 1
                    """,
                    (transaction_id,),
                )
            else:
                cursor.execute(
                    """
                    select decision_id, event_id, event_ts, final_score, decision, served_by, model_versions, scores, features, created_at
                    from decision_log
                    order by created_at desc
                    limit 1
                    """
                )
            row = cursor.fetchone()
    if not row:
        return {"status": "not_found"}
    return {
        "decision_id": row[0],
        "event_id": row[1],
        "event_ts": row[2].isoformat() if row[2] else None,
        "final_score": row[3],
        "decision": row[4],
        "served_by": row[5],
        "model_versions": row[6],
        "scores": row[7],
        "feature_snapshot": row[8],
        "created_at": row[9].isoformat() if row[9] else None,
    }


def _bundle_root(served_by: str) -> Path:
    env_path = os.getenv("MODEL_BUNDLE_PATH")
    if env_path:
        return Path(env_path)
    return Path("model_bundles") / served_by


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain a single transaction.")
    parser.add_argument("--transaction_id")
    parser.add_argument("--base_url", default="http://localhost:8083")
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    try:
        if args.transaction_id:
            decision = _fetch_json(f"{args.base_url}/debug/decision/{args.transaction_id}")
        else:
            decision = _fetch_json(f"{args.base_url}/debug/last-decision")
    except HTTPError:
        decision = _fetch_db(args.transaction_id)
    if decision.get("status") in {"disabled", "empty"}:
        raise SystemExit(f"Decision not available: {decision.get('status')}")
    if decision.get("status") == "not_found":
        decision = _fetch_db(None)
        if decision.get("status") in {"not_found", "empty"}:
            raise SystemExit("Decision not available: not_found")

    tx_id = decision.get("event_id")
    served_by = decision.get("served_by") or "v1"
    feature_snapshot = decision.get("feature_snapshot") or {}

    out_dir = Path("monitoring/reports/latest/local_explanations")
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / f"{tx_id}_features.json"
    features_path.write_text(json.dumps(feature_snapshot, indent=2), encoding="utf-8")

    bundle_root = _bundle_root(served_by)
    model_path = bundle_root / "gbdt" / "model.pkl"
    feature_order = bundle_root / "gbdt" / "feature_order.json"
    if not model_path.exists():
        model_path = Path("artifacts_phase4") / "gbdt" / "model.pkl"
        feature_order = Path("artifacts_phase4") / "gbdt" / "feature_order.json"
    if not feature_order.exists():
        feature_order = Path("artifacts_phase4") / "gbdt" / "feature_order.json"
    if not feature_order.exists():
        raise SystemExit("feature_order.json not found. Re-run train_phase4.py to generate it.")

    shap_out = out_dir / f"{tx_id}_shap.json"
    cmd = [
        "python",
        "src/demo/shap_for_event.py",
        "--model_path",
        str(model_path),
        "--features_json",
        str(features_path),
        "--feature_order_json",
        str(feature_order),
        "--top_k",
        str(args.top_k),
        "--output_json",
        str(shap_out),
    ]
    subprocess.run(cmd, check=True)

    event_path = out_dir / f"{tx_id}_event.json"
    event_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    slice_out = out_dir / f"{tx_id}_slice_context.json"
    subprocess.run(
        [
            "python",
            "src/demo/slice_context.py",
            "--event_json",
            str(event_path),
            "--output_json",
            str(slice_out),
        ],
        check=False,
    )

    print(json.dumps({"event_id": tx_id, "decision": decision.get("decision"), "score": decision.get("final_score")}, indent=2))
    print(f"SHAP: {shap_out}")
    print(f"Slice context: {slice_out}")
    latest_drift = Path("monitoring/reports/latest/evidently_drift.html")
    if latest_drift.exists():
        print(f"Drift report: {latest_drift}")


if __name__ == "__main__":
    main()
