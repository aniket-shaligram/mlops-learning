from __future__ import annotations

import argparse
import time
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for HTTP endpoint to be ready.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    start = time.time()
    while time.time() - start < args.timeout:
        try:
            with urlopen(args.url, timeout=2) as resp:
                if 200 <= resp.status < 500:
                    print(f"ready: {args.url}")
                    return
        except Exception:
            time.sleep(args.interval)
    raise SystemExit(f"timeout waiting for {args.url}")


if __name__ == "__main__":
    main()
