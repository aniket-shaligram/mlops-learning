from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.feedback import evaluate_performance
from src.utils import ensure_dir, save_json


def _parse_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _drift_detected(summary_path: Optional[str]) -> bool:
    if not summary_path:
        return False
    path = Path(summary_path)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    for key in ("input_drift", "feature_drift", "prediction_drift"):
        if data.get(key, {}).get("drift_detected"):
            return True
    return False


def _apply_pgdsn_env(env: Dict[str, str]) -> Dict[str, str]:
    dsn = env.get("PG_DSN") or os.getenv("PG_DSN")
    if not dsn:
        return env
    parsed = urlparse(dsn)
    if parsed.hostname:
        env.setdefault("POSTGRES_HOST", parsed.hostname)
    if parsed.port:
        env.setdefault("POSTGRES_PORT", str(parsed.port))
    if parsed.username:
        env.setdefault("POSTGRES_USER", parsed.username)
    if parsed.password:
        env.setdefault("POSTGRES_PASSWORD", parsed.password)
    if parsed.path and parsed.path.lstrip("/"):
        env.setdefault("POSTGRES_DB", parsed.path.lstrip("/"))
    return env


def run(
    as_of: str = "now",
    window_hours: int = 24,
    min_labeled_rows: int = 500,
    perf_floor_f1: float = 0.6,
    perf_floor_precision: float = 0.5,
    drift_flag_file: Optional[str] = None,
) -> Dict[str, Any]:
    as_of_dt = _parse_time(as_of)
    report_dir = ensure_dir(
        Path("feedback") / "reports" / "retrain" / as_of_dt.strftime("%Y%m%d_%H%M%S")
    )

    perf = evaluate_performance.evaluate(
        window_hours=window_hours,
        as_of=as_of,
        threshold=0.7,
    )
    drift_hit = _drift_detected(drift_flag_file)

    triggered = False
    reason = "ok"
    overall = perf.get("overall") if perf.get("status") == "ok" else None
    rows = perf.get("overall", {}).get("rows") if perf.get("status") == "ok" else perf.get("rows", 0)

    if perf.get("status") != "ok":
        reason = "insufficient_perf_data"
        triggered = False
    else:
        if rows < min_labeled_rows:
            reason = "min_labeled_rows_not_met"
            triggered = False
        else:
            f1 = overall.get("f1", 0.0) if overall else 0.0
            precision = overall.get("precision", 0.0) if overall else 0.0
            if f1 < perf_floor_f1 or precision < perf_floor_precision:
                triggered = True
                reason = "perf_floor_breached"
            if drift_hit:
                triggered = True
                reason = "drift_detected"

    trigger_report = {
        "triggered": triggered,
        "reason": reason,
        "perf": overall,
        "drift_detected": drift_hit,
        "min_labeled_rows": min_labeled_rows,
    }
    save_json(Path(report_dir) / "trigger.json", trigger_report)

    if not triggered:
        save_json(Path(report_dir) / "candidate.json", {"status": "no_action"})
        return trigger_report

    candidate_dir = Path("artifacts_phase4_candidate") / as_of_dt.strftime("%Y%m%d_%H%M%S")
    train_cmd = [
        "python",
        "src/train_phase4.py",
        "--start",
        "now-30d",
        "--end",
        "now",
        "--model_dir",
        str(candidate_dir),
        "--registered_model_name",
        "fraud-risk",
    ]
    env = os.environ.copy()
    env["USE_TXN_LABELS"] = "true"
    env = _apply_pgdsn_env(env)
    subprocess.run(train_cmd, check=True, env=env)

    candidate_meta = {
        "candidate_dir": str(candidate_dir),
        "training_window": {"start": "now-30d", "end": "now"},
        "perf_at_trigger": overall,
    }
    trigger_report["candidate_dir"] = str(candidate_dir)
    save_json(Path(report_dir) / "candidate.json", candidate_meta)
    return trigger_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain trigger based on perf/drift.")
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--window_hours", type=int, default=24)
    parser.add_argument("--min_labeled_rows", type=int, default=500)
    parser.add_argument("--perf_floor_f1", type=float, default=0.6)
    parser.add_argument("--perf_floor_precision", type=float, default=0.5)
    parser.add_argument("--drift_flag_file")
    args = parser.parse_args()

    result = run(
        as_of=args.as_of,
        window_hours=args.window_hours,
        min_labeled_rows=args.min_labeled_rows,
        perf_floor_f1=args.perf_floor_f1,
        perf_floor_precision=args.perf_floor_precision,
        drift_flag_file=args.drift_flag_file,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
