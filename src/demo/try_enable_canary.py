from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def main() -> None:
    parser = argparse.ArgumentParser(description="Try to enable canary/shadow at runtime.")
    parser.add_argument("--base_url", default="http://localhost:8080")
    parser.add_argument("--canary_percent", type=int, default=20)
    parser.add_argument("--shadow", action="store_true")
    args = parser.parse_args()

    body = json.dumps({"canary_percent": args.canary_percent, "shadow": args.shadow}).encode("utf-8")
    url = f"{args.base_url}/admin/traffic"
    try:
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as resp:
            print(resp.read().decode("utf-8"))
            return
    except Exception as exc:
        print(json.dumps({"status": "not_available", "error": str(exc)}))
        print("If unavailable, restart router with ROUTER_MODE/CANARY_PERCENT env vars.")


if __name__ == "__main__":
    main()
