from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.utils import ensure_dir, save_json


def run(candidate_dir: str, target: str = "v2") -> dict:
    candidate_path = Path(candidate_dir)
    if not candidate_path.exists():
        raise FileNotFoundError(f"Candidate dir not found: {candidate_dir}")

    bundles_root = Path("model_bundles")
    target_dir = bundles_root / target
    ensure_dir(target_dir)

    for item in candidate_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    bundle_meta = {
        "candidate_dir": str(candidate_path),
        "target": target,
    }
    save_json(target_dir / "bundle.json", bundle_meta)
    return bundle_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote candidate bundle to v2.")
    parser.add_argument("--candidate_dir", required=True)
    parser.add_argument("--target", default="v2")
    args = parser.parse_args()
    meta = run(candidate_dir=args.candidate_dir, target=args.target)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
