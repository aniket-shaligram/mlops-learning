from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


class RulesModel:
    def __init__(self, thresholds: Dict[str, float] | None = None) -> None:
        defaults = {
            "high_amount": 1500.0,
            "very_high_amount": 5000.0,
            "velocity_count_5m": 6.0,
            "velocity_amount_1h": 2500.0,
            "merchant_risk": 0.8,
            "base_risk": 0.05,
            "high_risk": 0.95,
            "elevated_risk": 0.75,
        }
        if thresholds:
            defaults.update(thresholds)
        self.thresholds = defaults

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        t = self.thresholds
        risk = np.full(len(features_df), t["base_risk"], dtype=float)

        amount = features_df.get("amount", pd.Series(0, index=features_df.index))
        geo_mismatch = features_df.get("geo_mismatch", pd.Series(0, index=features_df.index))
        is_new_device = features_df.get("is_new_device", pd.Series(0, index=features_df.index))
        is_new_ip = features_df.get("is_new_ip", pd.Series(0, index=features_df.index))

        suspicious_context = (geo_mismatch.astype(float) + is_new_device.astype(float) + is_new_ip.astype(float)) >= 1

        high_amount = amount >= t["high_amount"]
        very_high_amount = amount >= t["very_high_amount"]

        risk = np.where(very_high_amount & suspicious_context, t["high_risk"], risk)
        risk = np.where(high_amount & suspicious_context, np.maximum(risk, t["elevated_risk"]), risk)

        velocity_count = features_df.get("user_txn_count_5m", pd.Series(0, index=features_df.index))
        velocity_amount = features_df.get("user_amount_sum_1h", pd.Series(0, index=features_df.index))
        risk = np.where(velocity_count >= t["velocity_count_5m"], np.maximum(risk, t["elevated_risk"]), risk)
        risk = np.where(velocity_amount >= t["velocity_amount_1h"], np.maximum(risk, t["elevated_risk"]), risk)

        merchant_risk = features_df.get("merchant_chargeback_rate_30d", pd.Series(0, index=features_df.index))
        risk = np.where(merchant_risk >= t["merchant_risk"], np.maximum(risk, t["elevated_risk"]), risk)

        risk = np.clip(risk, 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])

