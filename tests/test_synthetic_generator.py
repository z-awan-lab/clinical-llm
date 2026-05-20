"""Tests for the synthetic data generator."""

import numpy as np
import pandas as pd
import pytest

from clinical_llm.data.synthetic_generator import (
    VITALS,
    GeneratorConfig,
    generate,
)


@pytest.fixture
def small_dataset():
    config = GeneratorConfig(n_patients=50, seed=123)
    return generate(config)


def test_generator_returns_two_dataframes(small_dataset):
    patients, events = small_dataset
    assert isinstance(patients, pd.DataFrame)
    assert isinstance(events, pd.DataFrame)


def test_patient_count_matches_config(small_dataset):
    patients, _ = small_dataset
    assert len(patients) == 50


def test_required_patient_columns_present(small_dataset):
    patients, _ = small_dataset
    required = {"patient_id", "age", "sex", "in_hospital_mortality"}
    assert required.issubset(patients.columns)


def test_required_event_columns_present(small_dataset):
    _, events = small_dataset
    required = {"patient_id", "charttime", "vital_name", "value", "unit"}
    assert required.issubset(events.columns)


def test_outcome_is_binary(small_dataset):
    patients, _ = small_dataset
    assert set(patients["in_hospital_mortality"].unique()).issubset({0, 1})


def test_ages_in_plausible_range(small_dataset):
    patients, _ = small_dataset
    assert patients["age"].min() >= 18
    assert patients["age"].max() <= 95


def test_all_vitals_appear(small_dataset):
    _, events = small_dataset
    assert set(events["vital_name"].unique()) == set(VITALS.keys())


def test_no_event_values_are_nan(small_dataset):
    # Missing values are dropped during generation, not represented as NaN rows.
    _, events = small_dataset
    assert not events["value"].isna().any()


def test_generator_is_deterministic():
    cfg = GeneratorConfig(n_patients=20, seed=7)
    p1, e1 = generate(cfg)
    p2, e2 = generate(cfg)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(e1, e2)


def test_different_seeds_give_different_data():
    p1, _ = generate(GeneratorConfig(n_patients=50, seed=1))
    p2, _ = generate(GeneratorConfig(n_patients=50, seed=2))
    # Outcomes should differ across seeds (with very high probability).
    assert not np.array_equal(
        p1["in_hospital_mortality"].values, p2["in_hospital_mortality"].values
    )
