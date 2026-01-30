from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _payload(rng: random.Random, mode: str) -> dict:
    user_id = rng.randint(1, 10000)
    amount = rng.random() * 5000
    drift_phase = 0
    country = rng.choice(["US", "GB", "CA", "DE", "IN", "SG", "AU", "FR", "JP"])
    channel = "mobile" if rng.random() > 0.5 else "web"
    if mode == "drift":
        drift_phase = 1
        amount = 1000 + rng.random() * 4000
        country = "NG"
        channel = "mobile"
    return {
        "event_id": f"evt_{int(time.time() * 1000)}_{rng.randint(1, 9999)}",
        "event_type": "txn.created",
        "event_ts": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "merchant_id": rng.randint(1, 5000),
        "device_id": rng.randint(1, 2000),
        "ip_id": rng.randint(1, 100000),
        "amount": round(amount, 2),
        "currency": "USD",
        "country": country,
        "channel": channel,
        "drift_phase": drift_phase,
    }


def _post(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=3) as resp:
            resp.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Send traffic to /score.")
    parser.add_argument("--url", default="http://localhost:8080/score")
    parser.add_argument("--qps", type=float, default=5.0)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--mode", default="baseline", choices=["baseline", "drift", "canary"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    total = int(args.qps * args.seconds)
    sleep = 1.0 / args.qps if args.qps > 0 else 0.0
    sent = 0
    for _ in range(total):
        payload = _payload(rng, "drift" if args.mode == "drift" else "baseline")
        for attempt in range(3):
            try:
                _post(args.url, payload)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                time.sleep(0.5)
        sent += 1
        if sleep:
            time.sleep(sleep)
    print(json.dumps({"sent": sent, "mode": args.mode, "url": args.url}, indent=2))


if __name__ == "__main__":
    main()
