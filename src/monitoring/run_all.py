from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.utils import ensure_dir
from src.monitoring import run_evidently, run_shap, run_slices


def main() -> None:
    parser = argparse.ArgumentParser(description="Run monitoring pipeline.")
    parser.add_argument("--ref_hours", type=int, default=24)
    parser.add_argument("--cur_hours", type=int, default=1)
    parser.add_argument("--as_of", default="now")
    parser.add_argument("--sample_size", type=int, default=500)
    args = parser.parse_args()

    run_evidently.run(ref_hours=args.ref_hours, cur_hours=args.cur_hours, as_of=args.as_of)
    run_shap.run(
        ref_hours=args.ref_hours,
        cur_hours=args.cur_hours,
        as_of=args.as_of,
        sample_size=args.sample_size,
    )
    run_slices.run(hours=args.ref_hours, as_of=args.as_of)

    index_dir = ensure_dir(Path("monitoring") / "reports")
    index_path = Path(index_dir) / "index.html"
    index_path.write_text(
        "<html><body><h2>Monitoring Reports</h2>"
        "<p>See subfolders under monitoring/reports for the latest outputs.</p>"
        "</body></html>",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
