"""Tests for evaluation metrics."""

import numpy as np
import pytest

from clinical_llm.evaluation.metrics import MetricWithCI, evaluate


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred = y_true.astype(float)  # exactly correct
    return y_true, y_pred


@pytest.fixture
def random_predictions():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.random(200)
    return y_true, y_pred


def test_perfect_predictor_gets_auroc_1(perfect_predictions):
    y_true, y_pred = perfect_predictions
    results = evaluate(y_true, y_pred, n_bootstrap=100)
    assert results.auroc.point == pytest.approx(1.0)


def test_perfect_predictor_gets_low_brier(perfect_predictions):
    y_true, y_pred = perfect_predictions
    results = evaluate(y_true, y_pred, n_bootstrap=100)
    assert results.brier.point == pytest.approx(0.0)


def test_random_predictor_auroc_near_half(random_predictions):
    y_true, y_pred = random_predictions
    results = evaluate(y_true, y_pred, n_bootstrap=200)
    # Random predictions should give AUROC near 0.5 (within bootstrap CI).
    assert 0.35 < results.auroc.point < 0.65


def test_ci_lower_le_point_le_upper(random_predictions):
    y_true, y_pred = random_predictions
    results = evaluate(y_true, y_pred, n_bootstrap=200)
    for metric in [results.auroc, results.auprc, results.brier]:
        assert metric.ci_lower <= metric.point <= metric.ci_upper, \
            f"CI not bracketing point estimate: {metric}"


def test_n_samples_correct(random_predictions):
    y_true, y_pred = random_predictions
    results = evaluate(y_true, y_pred, n_bootstrap=50)
    assert results.n_samples == len(y_true)
    assert results.n_positive == int(y_true.sum())


def test_metric_with_ci_str_format():
    m = MetricWithCI(point=0.812, ci_lower=0.780, ci_upper=0.844)
    assert "0.812" in str(m)
    assert "0.780" in str(m)
    assert "0.844" in str(m)


def test_evaluate_rejects_non_binary_labels():
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    with pytest.raises(AssertionError):
        evaluate(y_true, y_pred, n_bootstrap=10)
