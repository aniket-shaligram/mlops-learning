from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.feedback import evaluate_performance, labeler, promote_candidate, retrain_trigger
from src.utils import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feedback loop.")
    parser.add_argument("--delay_minutes", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--window_hours", type=int, default=24)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--drift_flag_file")
    args = parser.parse_args()

    as_of = args.as_of
    labeler.run(delay_minutes=args.delay_minutes, batch_size=args.batch_size, as_of=as_of)
    perf = evaluate_performance.run(
        window_hours=args.window_hours,
        as_of=as_of,
        threshold=args.threshold,
    )
    trigger = retrain_trigger.run(
        as_of=as_of,
        window_hours=args.window_hours,
        drift_flag_file=args.drift_flag_file,
    )

    if args.promote and trigger.get("triggered"):
        candidate_dir = trigger.get("candidate_dir")
        if candidate_dir and Path(candidate_dir).exists():
            promote_candidate.run(candidate_dir=str(candidate_dir), target="v2")

    index_dir = ensure_dir(Path("feedback") / "reports")
    index_path = Path(index_dir) / "index.html"
    index_path.write_text(
        "<html><body><h2>Feedback Loop Reports</h2>"
        "<p>See feedback/reports for labeler/perf/retrain outputs.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    print(json.dumps({"perf": perf.get("status"), "triggered": trigger.get("triggered")}, indent=2))


if __name__ == "__main__":
    main()
