"""Tests for the LSTM baseline.

These tests use a tiny LSTM (1 epoch, hidden size 8) so they run in
seconds. They check shapes, interfaces, and basic learning behaviour
rather than benchmark performance.
"""

import numpy as np
import pytest

from clinical_llm.data.sequences import (
    SequenceConfig,
    SequenceDataset,
    _build_pivot,
    compute_normalisation_stats,
)
from clinical_llm.data.splits import filter_events_by_patients, make_splits
from clinical_llm.data.synthetic_generator import GeneratorConfig, generate
from clinical_llm.models.lstm import LSTMBaseline, LSTMConfig


@pytest.fixture(scope="module")
def datasets():
    """Build small train/val/test sequence datasets from synthetic data."""
    cfg = GeneratorConfig(n_patients=120, seed=23)
    patients, events = generate(cfg)
    train_pts, val_pts, test_pts = make_splits(patients, seed=0)

    seq_cfg = SequenceConfig(observation_hours=48)
    train_events = filter_events_by_patients(events, train_pts)
    val_events = filter_events_by_patients(events, val_pts)
    test_events = filter_events_by_patients(events, test_pts)

    pivot = _build_pivot(train_events, seq_cfg)
    stats = compute_normalisation_stats(pivot)

    train_ds = SequenceDataset(train_pts, train_events, stats, config=seq_cfg)
    val_ds = SequenceDataset(val_pts, val_events, stats, config=seq_cfg)
    test_ds = SequenceDataset(test_pts, test_events, stats, config=seq_cfg)

    return train_ds, val_ds, test_ds


def _tiny_config(seed: int = 0) -> LSTMConfig:
    return LSTMConfig(
        hidden_size=8,
        num_layers=1,
        epochs=2,
        batch_size=16,
        learning_rate=1e-2,
        early_stopping_patience=2,
        seed=seed,
    )


def test_lstm_fits_and_predicts(datasets):
    train_ds, val_ds, _ = datasets
    model = LSTMBaseline(_tiny_config())
    model.fit(train_ds, val_ds)
    probs = model.predict_proba(val_ds)
    assert probs.shape == (len(val_ds),)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_lstm_raises_if_not_fit(datasets):
    _, val_ds, _ = datasets
    model = LSTMBaseline(_tiny_config())
    with pytest.raises(RuntimeError):
        model.predict_proba(val_ds)


def test_lstm_predictions_length_matches_test_set(datasets):
    train_ds, val_ds, test_ds = datasets
    model = LSTMBaseline(_tiny_config())
    model.fit(train_ds, val_ds)
    probs = model.predict_proba(test_ds)
    assert len(probs) == len(test_ds)


def test_lstm_is_deterministic(datasets):
    train_ds, val_ds, test_ds = datasets
    m1 = LSTMBaseline(_tiny_config(seed=7))
    m1.fit(train_ds, val_ds)
    p1 = m1.predict_proba(test_ds)

    m2 = LSTMBaseline(_tiny_config(seed=7))
    m2.fit(train_ds, val_ds)
    p2 = m2.predict_proba(test_ds)

    np.testing.assert_array_almost_equal(p1, p2, decimal=4)
