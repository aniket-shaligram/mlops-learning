from __future__ import annotations

from typing import Any, Dict, Tuple

from feast import FeatureStore

from src.feast_client import FEAST_FEATURE_REFS

FEATURE_SERVICE_NAME = "fraud_feature_service"


class OnlineFeatureClient:
    def __init__(self, repo_path: str = "feast_repo") -> None:
        self.repo_path = repo_path
        self._store: FeatureStore | None = None

    def _get_store(self) -> FeatureStore:
        if self._store is None:
            self._store = FeatureStore(repo_path=self.repo_path)
        return self._store

    def fetch_online_features(self, event: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        try:
            entity_row = {
                "user_id": event["user_id"],
                "merchant_id": event["merchant_id"],
                "device_id": event["device_id"],
                "ip_id": event["ip_id"],
            }
        except Exception:
            return {}, True

        try:
            store = self._get_store()
            try:
                response = store.get_online_features(
                    features=[FEATURE_SERVICE_NAME],
                    entity_rows=[entity_row],
                ).to_dict()
            except Exception:
                response = store.get_online_features(
                    features=FEAST_FEATURE_REFS,
                    entity_rows=[entity_row],
                ).to_dict()

            features: Dict[str, Any] = {}
            for key, value in response.items():
                if isinstance(value, list):
                    features[key] = value[0] if value else None
                else:
                    features[key] = value
            return features, False
        except Exception:
            return {}, True

    def health_check(self) -> Tuple[bool, str | None]:
        try:
            _ = self._get_store()
            return True, None
        except Exception as exc:
            return False, str(exc)
