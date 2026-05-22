"""Tests for serialisation of clinical sequences to LLM prompts.

These tests do not require transformers/peft/bitsandbytes — they test
the text serialisation logic, which is the part most likely to break
in subtle ways (off-by-one in time buckets, missing patients, etc.).
"""

import pandas as pd
import pytest

from clinical_llm.data.serialisation import (
    VITAL_CODES,
    SerialisationConfig,
    build_prompts,
    serialise_patient,
)
from clinical_llm.data.synthetic_generator import GeneratorConfig, generate


@pytest.fixture(scope="module")
def cohort():
    cfg = GeneratorConfig(n_patients=25, seed=42)
    return generate(cfg)


def test_serialise_patient_includes_age_and_sex(cohort):
    patients, events = cohort
    row = patients.iloc[0]
    pid = int(row["patient_id"])
    patient_events = events[events["patient_id"] == pid]
    prompt = serialise_patient(row, patient_events, SerialisationConfig())
    assert "Patient:" in prompt
    assert row["sex"] in prompt
    assert str(int(round(row["age"]))) in prompt


def test_serialise_patient_ends_with_outcome_marker(cohort):
    patients, events = cohort
    row = patients.iloc[0]
    pid = int(row["patient_id"])
    patient_events = events[events["patient_id"] == pid]
    prompt = serialise_patient(row, patient_events, SerialisationConfig())
    assert prompt.rstrip().endswith("Outcome:")


def test_serialise_patient_handles_no_events(cohort):
    patients, _ = cohort
    row = patients.iloc[0]
    empty_events = pd.DataFrame(columns=["patient_id", "charttime", "vital_name", "value", "unit"])
    prompt = serialise_patient(row, empty_events, SerialisationConfig())
    assert "no vital sign measurements" in prompt
    assert prompt.rstrip().endswith("Outcome:")


def test_serialise_uses_short_vital_codes(cohort):
    patients, events = cohort
    row = patients.iloc[0]
    pid = int(row["patient_id"])
    patient_events = events[events["patient_id"] == pid]
    prompt = serialise_patient(row, patient_events, SerialisationConfig())
    # At least one of the short codes should appear in the prompt.
    assert any(code in prompt for code in VITAL_CODES.values())


def test_serialise_respects_observation_window(cohort):
    patients, events = cohort
    row = patients.iloc[0]
    pid = int(row["patient_id"])
    patient_events = events[events["patient_id"] == pid]
    prompt = serialise_patient(row, patient_events, SerialisationConfig(observation_hours=24))
    # No row labelled Hour 24 or later should appear.
    for h in range(24, 48):
        assert f"Hour {h}:" not in prompt


def test_build_prompts_returns_one_per_patient(cohort):
    patients, events = cohort
    prompts = build_prompts(patients, events)
    assert len(prompts) == len(patients)


def test_build_prompts_handles_patient_with_no_events(cohort):
    patients, events = cohort
    # Drop all events for one patient and confirm the prompt is still built.
    dropped_pid = int(patients.iloc[0]["patient_id"])
    events_filtered = events[events["patient_id"] != dropped_pid]
    prompts = build_prompts(patients, events_filtered)
    assert len(prompts) == len(patients)
    # The first patient should have the "no vital sign measurements" prompt.
    assert "no vital sign measurements" in prompts[0]
