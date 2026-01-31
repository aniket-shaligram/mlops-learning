from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def _load_event(path: str) -> Dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_slices(path: str) -> List[Dict[str, str]]:
    rows = []
    if not Path(path).exists():
        return rows
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice context for one event.")
    parser.add_argument("--event_json", required=True)
    parser.add_argument("--slices_path", default="monitoring/reports/latest/slices.csv")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    event = _load_event(args.event_json)
    rows = _load_slices(args.slices_path)
    if not rows:
        out = {"context": [], "status": "slices_missing"}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps(out, indent=2))
        return

    context = []
    for dim in ("country", "channel", "drift_phase", "served_by"):
        value = event.get(dim)
        if value is None:
            continue
        for row in rows:
            if row.get("slice") == dim and row.get("value") == str(value):
                context.append(row)
                break

    out = {"context": context}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
