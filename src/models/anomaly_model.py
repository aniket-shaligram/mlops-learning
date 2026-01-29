from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    def __init__(
        self,
        contamination: float = 0.02,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.score_min: float = 0.0
        self.score_max: float = 1.0

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        if y is not None and y.nunique() > 1:
            train_X = X[y == 0]
            if train_X.empty:
                train_X = X
        else:
            train_X = X

        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self.model.fit(train_X)
        scores = self.model.decision_function(train_X)
        self.score_min = float(np.min(scores))
        self.score_max = float(np.max(scores))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        scores = self.model.decision_function(X)
        denom = self.score_max - self.score_min
        if denom <= 0:
            normalized = np.zeros_like(scores)
        else:
            normalized = (scores - self.score_min) / denom
        risk = 1.0 - normalized
        risk = np.clip(risk, 0.0, 1.0)
        return np.column_stack([1.0 - risk, risk])
