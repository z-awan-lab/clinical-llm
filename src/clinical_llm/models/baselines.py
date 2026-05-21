"""Tabular baselines for in-hospital mortality prediction.

This module provides two baselines that operate on aggregated, flat features:
a logistic regression and an XGBoost classifier. Both expose the same
fit / predict_proba interface so the training entry point treats them
uniformly.

These baselines exist to set an honest floor. If a fine-tuned LLM cannot
meaningfully beat them, that is the most important finding of the project
and should be reported transparently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
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


@dataclass
class XGBoostConfig:
    """Hyperparameters for the XGBoost baseline.

    Defaults are deliberately mild — strong enough to give a fair fight,
    not so tuned that they obscure the methodological point of the
    comparison. Real benchmarking would sweep these on the validation set.
    """

    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 1.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    seed: int = 42


class XGBoostBaseline:
    """Gradient-boosted trees on aggregated clinical features.

    Operates on the same flat feature matrix as the logistic regression
    baseline. Handles class imbalance via ``scale_pos_weight``, computed
    from the training set's class frequencies. Uses the validation set
    for early stopping to guard against overfitting on small cohorts.
    """

    def __init__(self, config: XGBoostConfig | None = None) -> None:
        self.config = config or XGBoostConfig()
        self.model_: xgb.XGBClassifier | None = None
        self.imputer_: SimpleImputer | None = None
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> XGBoostBaseline:
        self.feature_names_ = list(X.columns)

        # Median imputation, just like the logistic regression pipeline,
        # so the two baselines see identical inputs.
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X.values)

        # Compute scale_pos_weight from the training label distribution.
        n_neg = int((y == 0).sum())
        n_pos = int((y == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        eval_set = None
        early_stopping = None
        if X_val is not None and y_val is not None:
            X_val_imp = self.imputer_.transform(X_val.values)
            eval_set = [(X_val_imp, y_val)]
            early_stopping = self.config.early_stopping_rounds

        self.model_ = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            random_state=self.config.seed,
            early_stopping_rounds=early_stopping,
            verbosity=0,
        )
        self.model_.fit(X_imp, y, eval_set=eval_set, verbose=False)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.imputer_ is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        X_ordered = X[self.feature_names_].values
        X_imp = self.imputer_.transform(X_ordered)
        return self.model_.predict_proba(X_imp)[:, 1]

    @property
    def feature_importances(self) -> pd.Series:
        """Return feature importances by gain, sorted descending."""
        if self.model_ is None:
            raise RuntimeError("Model must be fit before reading feature importances.")
        importances = self.model_.feature_importances_
        return pd.Series(importances, index=self.feature_names_).sort_values(ascending=False)
