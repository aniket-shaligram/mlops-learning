from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.request import urlopen


def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=3) as resp:
        return json.loads(resp.read().decode("utf-8"))


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

    if args.transaction_id:
        decision = _fetch_json(f"{args.base_url}/debug/decision/{args.transaction_id}")
    else:
        decision = _fetch_json(f"{args.base_url}/debug/last-decision")
    if decision.get("status") in {"disabled", "not_found", "empty"}:
        raise SystemExit(f"Decision not available: {decision.get('status')}")

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
        check=True,
    )

    print(json.dumps({"event_id": tx_id, "decision": decision.get("decision"), "score": decision.get("final_score")}, indent=2))
    print(f"SHAP: {shap_out}")
    print(f"Slice context: {slice_out}")
    latest_drift = Path("monitoring/reports/latest/evidently_drift.html")
    if latest_drift.exists():
        print(f"Drift report: {latest_drift}")


if __name__ == "__main__":
    main()
