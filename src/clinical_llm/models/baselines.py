"""Logistic regression baseline.

A deliberately simple model. If a fine-tuned LLM cannot meaningfully beat
this, that is the most important finding of the project and should be
reported honestly. Recruiters and reviewers respect rigour over hype.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class LogRegConfig:
    C: float = 1.0
    max_iter: int = 1000
    class_weight: str | None = "balanced"
    seed: int = 42


class LogisticRegressionBaseline:
    """A standardised, median-imputed, class-balanced logistic regression.

    Wraps sklearn pieces into a single fit/predict_proba interface that the
    training script can call uniformly across all baselines.
    """

    def __init__(self, config: LogRegConfig | None = None) -> None:
        self.config = config or LogRegConfig()
        self.pipeline: Pipeline | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> LogisticRegressionBaseline:
        self.feature_names_ = list(X.columns)
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.C,
                        max_iter=self.config.max_iter,
                        class_weight=self.config.class_weight,
                        random_state=self.config.seed,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X.values, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        # Reorder columns to match training order, in case the caller scrambled them.
        X_ordered = X[self.feature_names_].values
        return self.pipeline.predict_proba(X_ordered)[:, 1]

    @property
    def coefficients(self) -> pd.Series:
        """Return logistic-regression coefficients keyed by feature name.

        Useful for simple interpretability before SHAP comes into play.
        """
        if self.pipeline is None:
            raise RuntimeError("Model must be fit before reading coefficients.")
        coefs = self.pipeline.named_steps["clf"].coef_.ravel()
        return pd.Series(coefs, index=self.feature_names_).sort_values(key=abs, ascending=False)
