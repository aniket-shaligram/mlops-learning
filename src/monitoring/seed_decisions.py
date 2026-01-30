from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from datetime import datetime, timezone


def _random_payload() -> dict:
    countries = ["US", "GB", "CA", "DE", "IN", "SG", "AU", "FR", "JP"]
    return {
        "event_id": f"evt_{int(time.time() * 1000)}_{random.randint(1, 9999)}",
        "event_type": "txn.created",
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "user_id": random.randint(1, 10000),
        "merchant_id": random.randint(1, 5000),
        "device_id": random.randint(1, 2000),
        "ip_id": random.randint(1, 100000),
        "amount": round(random.random() * 5000, 2),
        "currency": "USD",
        "country": random.choice(countries),
        "channel": "mobile" if random.random() > 0.5 else "web",
        "drift_phase": 0,
    }


def _post(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        resp.read()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate score traffic.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--url", default="http://localhost:8080/score")
    parser.add_argument("--sleep_ms", type=int, default=10)
    args = parser.parse_args()

    for _ in range(args.count):
        _post(args.url, _random_payload())
        if args.sleep_ms:
            time.sleep(args.sleep_ms / 1000.0)


if __name__ == "__main__":
    main()
