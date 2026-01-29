from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

SCRIPT_DIR = Path(__file__).resolve()
sys.path.append(str(SCRIPT_DIR.parents[1]))

import numpy as np
import pandas as pd

from utils import SYNTH_FEATURES


def _country_currency_map() -> Tuple[List[str], List[str], np.ndarray]:
    countries = ["US", "GB", "CA", "DE", "IN", "BR", "SG", "AU", "FR", "JP"]
    currencies = ["USD", "GBP", "CAD", "EUR", "INR", "BRL", "SGD", "AUD", "EUR", "JPY"]
    weights = np.array([0.28, 0.08, 0.06, 0.08, 0.16, 0.07, 0.05, 0.06, 0.08, 0.08])
    return countries, currencies, weights / weights.sum()


def _build_profiles(
    rng: np.random.Generator, num_users: int, num_merchants: int, num_devices: int, num_ips: int
) -> Dict[str, np.ndarray]:
    countries, _, country_weights = _country_currency_map()
    user_home_country = rng.choice(countries, size=num_users, p=country_weights)
    user_amount_mean_log = rng.normal(loc=3.2, scale=0.5, size=num_users)
    user_amount_std_log = rng.uniform(0.3, 0.7, size=num_users)
    user_activity_rate = rng.uniform(0.4, 3.0, size=num_users)
    user_baseline_risk = rng.uniform(0.0, 0.08, size=num_users)
    user_avg_amount_30d = rng.lognormal(user_amount_mean_log, user_amount_std_log)

    merchant_chargeback_rate = rng.uniform(0.0, 0.03, size=num_merchants)
    merchant_risk_tier = np.digitize(merchant_chargeback_rate, bins=[0.01, 0.02])
    merchant_amount_scale = rng.normal(loc=0.0, scale=0.4, size=num_merchants)

    device_risk_score = rng.uniform(0.0, 0.4, size=num_devices)
    ip_risk_score = rng.uniform(0.0, 0.4, size=num_ips)

    return {
        "user_home_country": user_home_country,
        "user_amount_mean_log": user_amount_mean_log,
        "user_amount_std_log": user_amount_std_log,
        "user_activity_rate": user_activity_rate,
        "user_baseline_risk": user_baseline_risk,
        "user_avg_amount_30d": user_avg_amount_30d,
        "merchant_chargeback_rate": merchant_chargeback_rate,
        "merchant_risk_tier": merchant_risk_tier,
        "merchant_amount_scale": merchant_amount_scale,
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score,
    }


def _write_profiles(
    output_dir: Path, profiles: Dict[str, np.ndarray]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "user_id": np.arange(len(profiles["user_home_country"])),
            "home_country": profiles["user_home_country"],
            "avg_amount_30d": profiles["user_avg_amount_30d"],
            "baseline_activity_rate": profiles["user_activity_rate"],
            "baseline_risk": profiles["user_baseline_risk"],
        }
    ).to_csv(output_dir / "users.csv", index=False)
    pd.DataFrame(
        {
            "merchant_id": np.arange(len(profiles["merchant_chargeback_rate"])),
            "merchant_chargeback_rate": profiles["merchant_chargeback_rate"],
            "merchant_risk_tier": profiles["merchant_risk_tier"],
        }
    ).to_csv(output_dir / "merchants.csv", index=False)
    pd.DataFrame(
        {
            "device_id": np.arange(len(profiles["device_risk_score"])),
            "device_risk_score": profiles["device_risk_score"],
        }
    ).to_csv(output_dir / "devices.csv", index=False)
    pd.DataFrame(
        {
            "ip_id": np.arange(len(profiles["ip_risk_score"])),
            "ip_risk_score": profiles["ip_risk_score"],
        }
    ).to_csv(output_dir / "ips.csv", index=False)


def _generate_day(
    rng: np.random.Generator,
    day_index: int,
    rows_for_day: int,
    start_epoch: int,
    fraud_rate: float,
    drift_start_day: int,
    profiles: Dict[str, np.ndarray],
    countries: List[str],
    currencies: List[str],
    country_weights: np.ndarray,
    campaign_merchants: np.ndarray,
    drift_merchants: np.ndarray,
) -> pd.DataFrame:
    hour_weights = np.array(
        [0.3, 0.2, 0.2, 0.2, 0.3, 0.6, 1.2, 1.6, 1.8, 1.7, 1.6, 1.5,
         1.4, 1.3, 1.4, 1.6, 1.8, 1.7, 1.4, 1.1, 0.8, 0.6, 0.5, 0.4]
    )
    hour_weights = hour_weights / hour_weights.sum()

    hours = rng.choice(24, size=rows_for_day, p=hour_weights)
    minutes = rng.integers(0, 60, size=rows_for_day)
    seconds = rng.integers(0, 60, size=rows_for_day)
    event_ts = start_epoch + day_index * 86400 + hours * 3600 + minutes * 60 + seconds
    drift_phase = int(day_index >= drift_start_day)

    num_users = len(profiles["user_home_country"])
    num_merchants = len(profiles["merchant_chargeback_rate"])
    num_devices = len(profiles["device_risk_score"])
    num_ips = len(profiles["ip_risk_score"])

    user_id = rng.integers(0, num_users, size=rows_for_day)
    merchant_id = rng.integers(0, num_merchants, size=rows_for_day)
    device_id = rng.integers(0, num_devices, size=rows_for_day)
    ip_id = rng.integers(0, num_ips, size=rows_for_day)

    user_home = profiles["user_home_country"][user_id]
    user_activity = profiles["user_activity_rate"][user_id]
    user_avg_amount = profiles["user_avg_amount_30d"][user_id]
    user_mean_log = profiles["user_amount_mean_log"][user_id]
    user_std_log = profiles["user_amount_std_log"][user_id]
    merchant_cb = profiles["merchant_chargeback_rate"][merchant_id]
    merchant_tier = profiles["merchant_risk_tier"][merchant_id]
    merchant_amount_scale = profiles["merchant_amount_scale"][merchant_id]

    base_country_weights = country_weights.copy()
    geo_drift_country = "NG"
    if drift_phase:
        countries = countries + [geo_drift_country]
        currencies = currencies + ["NGN"]
        base_country_weights = np.append(base_country_weights, 0.06)
        base_country_weights = base_country_weights / base_country_weights.sum()

    cross_border_rate = 0.02 if not drift_phase else 0.08
    cross_border_mask = rng.random(rows_for_day) < cross_border_rate
    txn_country = rng.choice(countries, size=rows_for_day, p=base_country_weights)
    txn_country = np.where(cross_border_mask, txn_country, user_home)
    currency_map = {country: currency for country, currency in zip(countries, currencies)}
    currency = pd.Series(txn_country).map(currency_map).values

    channel = rng.choice(["web", "mobile"], size=rows_for_day, p=[0.45, 0.55])
    base_new_device_rate = 0.02 if not drift_phase else 0.05
    base_new_ip_rate = 0.03 if not drift_phase else 0.06
    is_new_device = (rng.random(rows_for_day) < base_new_device_rate).astype(int)
    is_new_ip = (rng.random(rows_for_day) < base_new_ip_rate).astype(int)
    geo_mismatch = (txn_country != user_home).astype(int)

    base_amount = rng.lognormal(user_mean_log + merchant_amount_scale, user_std_log)
    base_amount = np.clip(base_amount, 0.5, 5000.0)
    amount = base_amount.copy()

    user_txn_count_5m = rng.poisson(user_activity)
    user_txn_count_1h = user_txn_count_5m + rng.poisson(user_activity * 6.0)

    device_risk_score = profiles["device_risk_score"][device_id] + rng.uniform(0.0, 0.1, size=rows_for_day)
    ip_risk_score = profiles["ip_risk_score"][ip_id] + rng.uniform(0.0, 0.1, size=rows_for_day)

    is_fraud = rng.random(rows_for_day) < fraud_rate
    fraud_mode = np.full(rows_for_day, -1)
    fraud_indices = np.where(is_fraud)[0]
    if fraud_indices.size > 0:
        fraud_mode[fraud_indices] = rng.choice([0, 1, 2], size=fraud_indices.size, p=[0.5, 0.3, 0.2])

    ato_mask = fraud_mode == 0
    if ato_mask.any():
        is_new_device[ato_mask] = 1 if not drift_phase else rng.integers(0, 2, size=ato_mask.sum())
        is_new_ip[ato_mask] = 1
        geo_mismatch[ato_mask] = 1
        amount[ato_mask] = amount[ato_mask] * rng.uniform(2.0, 5.0, size=ato_mask.sum())
        user_txn_count_5m[ato_mask] = rng.integers(3, 15, size=ato_mask.sum())
        user_txn_count_1h[ato_mask] = rng.integers(10, 40, size=ato_mask.sum())
        ip_risk_score[ato_mask] += 0.2
        device_risk_score[ato_mask] += 0.2

    card_mask = fraud_mode == 1
    if card_mask.any():
        is_new_device[card_mask] = rng.integers(0, 2, size=card_mask.sum())
        is_new_ip[card_mask] = rng.integers(0, 2, size=card_mask.sum())
        geo_mismatch[card_mask] = 0
        amount[card_mask] = rng.uniform(1.0, 20.0, size=card_mask.sum())
        user_txn_count_5m[card_mask] = rng.integers(30, 200, size=card_mask.sum())
        user_txn_count_1h[card_mask] = rng.integers(100, 500, size=card_mask.sum())
        ip_risk_score[card_mask] += 0.3

    campaign_mask = fraud_mode == 2
    if campaign_mask.any():
        merchant_id[campaign_mask] = rng.choice(campaign_merchants, size=campaign_mask.sum())
        merchant_cb[campaign_mask] = rng.uniform(0.08, 0.25, size=campaign_mask.sum())
        is_new_device[campaign_mask] = rng.integers(0, 2, size=campaign_mask.sum())
        is_new_ip[campaign_mask] = rng.integers(0, 2, size=campaign_mask.sum())
        geo_mismatch[campaign_mask] = rng.integers(0, 2, size=campaign_mask.sum())
        amount[campaign_mask] = rng.uniform(40.0, 600.0, size=campaign_mask.sum())
        user_txn_count_5m[campaign_mask] = rng.integers(2, 12, size=campaign_mask.sum())
        user_txn_count_1h[campaign_mask] = rng.integers(5, 40, size=campaign_mask.sum())

    if drift_phase:
        ip_risk_score[is_fraud] += 0.2
        merchant_drift_mask = np.isin(merchant_id, drift_merchants)
        merchant_cb[merchant_drift_mask] += 0.08

    user_amount_sum_1h = user_txn_count_1h * (
        amount * rng.uniform(0.7, 1.3, size=rows_for_day)
    )

    distance_from_home_km = np.where(
        geo_mismatch == 1,
        rng.uniform(500.0, 5000.0, size=rows_for_day),
        rng.exponential(30.0, size=rows_for_day),
    )

    delay_days = np.where(
        is_fraud, rng.integers(2, 31, size=rows_for_day), rng.integers(0, 2, size=rows_for_day)
    )
    label_available_ts = event_ts + delay_days * 86400 + rng.integers(0, 3600, size=rows_for_day)

    order = np.argsort(event_ts)
    event_ts = event_ts[order]
    user_id = user_id[order]
    merchant_id = merchant_id[order]
    device_id = device_id[order]
    ip_id = ip_id[order]
    amount = amount[order]
    currency = currency[order]
    txn_country = txn_country[order]
    channel = channel[order]
    hours = hours[order]
    is_new_device = is_new_device[order]
    is_new_ip = is_new_ip[order]
    distance_from_home_km = distance_from_home_km[order]
    geo_mismatch = geo_mismatch[order]
    user_txn_count_5m = user_txn_count_5m[order]
    user_txn_count_1h = user_txn_count_1h[order]
    user_amount_sum_1h = user_amount_sum_1h[order]
    user_avg_amount = user_avg_amount[order]
    merchant_cb = merchant_cb[order]
    device_risk_score = device_risk_score[order]
    ip_risk_score = ip_risk_score[order]
    label_available_ts = label_available_ts[order]
    is_fraud = is_fraud[order]

    df = pd.DataFrame(
        {
            "txn_id": np.arange(rows_for_day),
            "event_ts": pd.to_datetime(event_ts, unit="s", utc=True),
            "user_id": user_id,
            "merchant_id": merchant_id,
            "device_id": device_id,
            "ip_id": ip_id,
            "amount": amount,
            "currency": currency,
            "country": txn_country,
            "channel": channel,
            "hour_of_day": hours,
            "is_new_device": is_new_device,
            "is_new_ip": is_new_ip,
            "distance_from_home_km": distance_from_home_km,
            "geo_mismatch": geo_mismatch,
            "user_txn_count_5m": user_txn_count_5m,
            "user_txn_count_1h": user_txn_count_1h,
            "user_amount_sum_1h": user_amount_sum_1h,
            "user_avg_amount_30d": user_avg_amount,
            "merchant_chargeback_rate_30d": merchant_cb,
            "device_risk_score": np.clip(device_risk_score, 0.0, 1.0),
            "ip_risk_score": np.clip(ip_risk_score, 0.0, 1.0),
            "drift_phase": drift_phase,
            "is_fraud": is_fraud.astype(int),
            "label_available_ts": pd.to_datetime(label_available_ts, unit="s", utc=True),
        }
    )
    return df


def _write_chunk(
    df: pd.DataFrame,
    output_path: Path,
    fmt: str,
    is_first: bool,
    writer_state: Dict[str, object],
) -> None:
    if fmt == "csv":
        df.to_csv(output_path, mode="w" if is_first else "a", header=is_first, index=False)
        return

    if fmt == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise ImportError(
                "pyarrow is required for Parquet output. Install with pip install pyarrow."
            ) from exc

        table = pa.Table.from_pandas(df)
        if "writer" not in writer_state:
            writer_state["writer"] = pq.ParquetWriter(output_path, table.schema)
        writer = writer_state["writer"]
        writer.write_table(table)
        return

    raise ValueError(f"Unsupported format: {fmt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic fraud dataset.")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fraud_rate", type=float, default=0.01)
    parser.add_argument("--drift_start_day", type=int, default=20)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    rows = 10_000 if args.smoke_test else args.rows
    rng = np.random.default_rng(args.seed)
    output_path = Path(args.output_path or f"data/synth_transactions.{args.format}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_users = max(10_000, rows // 20)
    num_merchants = max(2_000, rows // 200)
    num_devices = int(num_users * 1.2)
    num_ips = int(num_users * 1.5)

    profiles = _build_profiles(rng, num_users, num_merchants, num_devices, num_ips)
    _write_profiles(output_path.parent / "synth_profiles", profiles)

    countries, currencies, country_weights = _country_currency_map()
    high_risk_merchants = np.where(profiles["merchant_risk_tier"] == 2)[0]
    if high_risk_merchants.size == 0:
        high_risk_merchants = np.arange(min(50, num_merchants))
    campaign_merchants = rng.choice(high_risk_merchants, size=min(200, len(high_risk_merchants)), replace=False)
    drift_merchants = rng.choice(high_risk_merchants, size=min(300, len(high_risk_merchants)), replace=False)

    rows_per_day = rows // args.days
    remainder = rows % args.days
    writer_state: Dict[str, object] = {}

    sample_written = False
    start_epoch = int(pd.Timestamp("2024-01-01").timestamp())
    txn_id_offset = 0

    for day in range(args.days):
        rows_for_day = rows_per_day + (1 if day < remainder else 0)
        if rows_for_day == 0:
            continue
        df_day = _generate_day(
            rng,
            day,
            rows_for_day,
            start_epoch,
            args.fraud_rate,
            args.drift_start_day,
            profiles,
            countries,
            currencies,
            country_weights,
            campaign_merchants,
            drift_merchants,
        )
        df_day["txn_id"] = np.arange(txn_id_offset, txn_id_offset + rows_for_day)
        txn_id_offset += rows_for_day

        if not sample_written:
            sample_columns = [
                "user_id",
                "merchant_id",
                "device_id",
                "ip_id",
                *SYNTH_FEATURES,
            ]
            sample = df_day[sample_columns].iloc[0].to_dict()
            sample_path = Path("examples/one_txn.json")
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_path.open("w", encoding="utf-8") as handle:
                json.dump(sample, handle, indent=2)
            sample_written = True

        _write_chunk(df_day, output_path, args.format, day == 0, writer_state)

    if args.format == "parquet" and "writer" in writer_state:
        writer_state["writer"].close()

    print(
        json.dumps(
            {"rows": rows, "output_path": str(output_path), "format": args.format},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
