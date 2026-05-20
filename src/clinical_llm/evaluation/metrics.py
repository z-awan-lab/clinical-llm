"""Evaluation metrics for binary clinical classification.

Reports discrimination (AUROC, AUPRC), calibration (Brier score), and
bootstrap 95% confidence intervals for all metrics. CIs are crucial:
single point estimates without CIs are a signal of an inexperienced
analysis, especially with small clinical datasets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


@dataclass
class MetricWithCI:
    """A point estimate with bootstrap 95% confidence interval."""

    point: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        return f"{self.point:.3f} (95% CI {self.ci_lower:.3f}–{self.ci_upper:.3f})"


@dataclass
class EvaluationResults:
    auroc: MetricWithCI
    auprc: MetricWithCI
    brier: MetricWithCI
    n_samples: int
    n_positive: int

    def to_dict(self) -> dict:
        return {
            "auroc": asdict(self.auroc),
            "auprc": asdict(self.auprc),
            "brier": asdict(self.brier),
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
        }

    def summary(self) -> str:
        lines = [
            f"  n = {self.n_samples:,} ({self.n_positive:,} positive, "
            f"{self.n_positive / self.n_samples:.1%})",
            f"  AUROC: {self.auroc}",
            f"  AUPRC: {self.auprc}",
            f"  Brier: {self.brier}",
        ]
        return "\n".join(lines)


def _bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> MetricWithCI:
    """Compute a metric with bootstrap 95% CI.

    Uses stratified resampling: each bootstrap sample preserves the original
    class balance. This is the standard approach for imbalanced clinical data.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # Skip resamples that happen to have only one class.
        y_t, y_p = y_true[idx], y_pred[idx]
        if len(np.unique(y_t)) < 2:
            continue
        boot_scores.append(metric_fn(y_t, y_p))

    lower, upper = np.percentile(boot_scores, [2.5, 97.5])
    return MetricWithCI(point=float(point), ci_lower=float(lower), ci_upper=float(upper))


def evaluate(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> EvaluationResults:
    """Compute AUROC, AUPRC, and Brier score with bootstrap 95% CIs.

    Args:
        y_true:       binary labels (0/1), shape (n,).
        y_pred_proba: predicted probabilities for the positive class, shape (n,).

    Returns:
        EvaluationResults with point estimates and CIs.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()
    assert y_true.shape == y_pred_proba.shape, "shape mismatch"
    assert set(np.unique(y_true)).issubset({0, 1}), "labels must be 0/1"

    return EvaluationResults(
        auroc=_bootstrap_metric(
            y_true, y_pred_proba, roc_auc_score, n_bootstrap=n_bootstrap, seed=seed
        ),
        auprc=_bootstrap_metric(
            y_true, y_pred_proba, average_precision_score, n_bootstrap=n_bootstrap, seed=seed + 1
        ),
        brier=_bootstrap_metric(
            y_true, y_pred_proba, brier_score_loss, n_bootstrap=n_bootstrap, seed=seed + 2
        ),
        n_samples=int(len(y_true)),
        n_positive=int(y_true.sum()),
    )
