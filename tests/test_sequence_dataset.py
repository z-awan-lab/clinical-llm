"""Tests for the sequence dataset used by sequence models."""

import numpy as np
import pytest
import torch

from clinical_llm.data.sequences import (
    SequenceConfig,
    SequenceDataset,
    _build_pivot,
    compute_normalisation_stats,
    n_features,
)
from clinical_llm.data.synthetic_generator import GeneratorConfig, generate


@pytest.fixture(scope="module")
def small_cohort():
    cfg = GeneratorConfig(n_patients=30, seed=11)
    return generate(cfg)


def test_pivot_has_correct_shape(small_cohort):
    _, events = small_cohort
    pivot = _build_pivot(events, SequenceConfig(observation_hours=48))
    # Each patient should have a (48, n_vitals) matrix.
    for df in pivot.values():
        assert df.shape == (48, n_features())


def test_normalisation_stats_shape(small_cohort):
    _, events = small_cohort
    pivot = _build_pivot(events, SequenceConfig(observation_hours=48))
    stats = compute_normalisation_stats(pivot)
    assert stats["mean"].shape == (n_features(),)
    assert stats["std"].shape == (n_features(),)
    assert not np.isnan(stats["mean"]).any()
    assert not np.isnan(stats["std"]).any()


def test_dataset_lengths_and_dtypes(small_cohort):
    patients, events = small_cohort
    cfg = SequenceConfig(observation_hours=48)
    pivot = _build_pivot(events, cfg)
    stats = compute_normalisation_stats(pivot)
    ds = SequenceDataset(patients, events, stats, config=cfg)

    assert len(ds) == len(patients)

    values, mask, label = ds[0]
    assert values.shape == (48, n_features())
    assert mask.shape == (48, n_features())
    assert values.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert label.dtype == torch.float32


def test_mask_is_binary(small_cohort):
    patients, events = small_cohort
    cfg = SequenceConfig(observation_hours=48)
    pivot = _build_pivot(events, cfg)
    stats = compute_normalisation_stats(pivot)
    ds = SequenceDataset(patients, events, stats, config=cfg)

    for i in range(len(ds)):
        _, mask, _ = ds[i]
        unique = torch.unique(mask).tolist()
        assert set(unique).issubset({0.0, 1.0})


def test_missing_positions_have_zero_values(small_cohort):
    patients, events = small_cohort
    cfg = SequenceConfig(observation_hours=48)
    pivot = _build_pivot(events, cfg)
    stats = compute_normalisation_stats(pivot)
    ds = SequenceDataset(patients, events, stats, config=cfg)

    for i in range(min(5, len(ds))):
        values, mask, _ = ds[i]
        # Wherever mask is zero, values should also be zero.
        missing_positions = mask == 0
        assert torch.all(values[missing_positions] == 0)
