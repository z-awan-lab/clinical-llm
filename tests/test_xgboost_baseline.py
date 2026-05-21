"""Tests for the XGBoost baseline."""

import numpy as np
import pandas as pd
import pytest

from clinical_llm.models.baselines import XGBoostBaseline, XGBoostConfig


@pytest.fixture
def toy_dataset():
    """A small, linearly separable dataset for sanity checks."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    # Outcome depends primarily on column 'a' — easy to learn.
    logits = 2.0 * X["a"].values + 0.5 * X["b"].values + rng.normal(scale=0.3, size=n)
    y = (logits > 0).astype(int)
    return X, y


def _split(X, y, frac=0.7):
    n_train = int(len(X) * frac)
    return X.iloc[:n_train], X.iloc[n_train:], y[:n_train], y[n_train:]


def test_xgb_fits_and_predicts_in_unit_interval(toy_dataset):
    X, y = toy_dataset
    X_tr, X_val, y_tr, y_val = _split(X, y)
    model = XGBoostBaseline(XGBoostConfig(n_estimators=50, seed=0))
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    p = model.predict_proba(X_val)
    assert p.shape == (len(X_val),)
    assert ((p >= 0) & (p <= 1)).all()


def test_xgb_beats_chance_on_easy_problem(toy_dataset):
    from sklearn.metrics import roc_auc_score

    X, y = toy_dataset
    X_tr, X_val, y_tr, y_val = _split(X, y)
    model = XGBoostBaseline(XGBoostConfig(n_estimators=100, seed=0))
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    auroc = roc_auc_score(y_val, model.predict_proba(X_val))
    assert auroc > 0.8, f"XGBoost did not learn the toy problem: AUROC={auroc:.3f}"


def test_xgb_raises_if_not_fit(toy_dataset):
    X, _ = toy_dataset
    model = XGBoostBaseline()
    with pytest.raises(RuntimeError):
        model.predict_proba(X)


def test_xgb_feature_importances_match_input_columns(toy_dataset):
    X, y = toy_dataset
    X_tr, X_val, y_tr, y_val = _split(X, y)
    model = XGBoostBaseline(XGBoostConfig(n_estimators=20, seed=0))
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val)
    imp = model.feature_importances
    assert set(imp.index) == set(X.columns)
    assert (imp >= 0).all()


def test_xgb_is_deterministic(toy_dataset):
    X, y = toy_dataset
    X_tr, X_val, y_tr, y_val = _split(X, y)
    m1 = XGBoostBaseline(XGBoostConfig(n_estimators=30, seed=7)).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val
    )
    m2 = XGBoostBaseline(XGBoostConfig(n_estimators=30, seed=7)).fit(
        X_tr, y_tr, X_val=X_val, y_val=y_val
    )
    np.testing.assert_array_almost_equal(m1.predict_proba(X_val), m2.predict_proba(X_val))
