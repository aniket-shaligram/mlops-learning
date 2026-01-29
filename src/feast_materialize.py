from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timedelta, timezone

import pandas as pd


def _parse_time(value: str) -> datetime:
    raw = value.strip().lower()
    now = datetime.now(timezone.utc)
    if raw == "now":
        return now

    match = re.match(r"^now([+-]\d+)([smhdw])$", raw)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        delta_args = {
            "s": "seconds",
            "m": "minutes",
            "h": "hours",
            "d": "days",
            "w": "weeks",
        }
        return now + timedelta(**{delta_args[unit]: amount})

    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Unable to parse datetime: {value}")
    return parsed.to_pydatetime()


def _run(cmd: list[str], cwd: str) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize Feast features to Redis.")
    parser.add_argument("--start", required=True, help='Start time (ISO or "now-1d").')
    parser.add_argument("--end", default="now", help='End time (ISO or "now").')
    parser.add_argument("--repo_path", default="feast_repo", help="Path to Feast repo.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Run `feast apply` before materialization.",
    )
    args = parser.parse_args()

    start = _parse_time(args.start)
    end = _parse_time(args.end)
    if end < start:
        raise ValueError("--end must be after --start")

    start_str = start.isoformat()
    end_str = end.isoformat()

    if args.apply:
        _run(["feast", "apply"], cwd=args.repo_path)
    _run(["feast", "materialize", start_str, end_str], cwd=args.repo_path)

    print(f"materialized features from {start_str} to {end_str}")


if __name__ == "__main__":
    main()
