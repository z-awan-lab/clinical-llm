"""Tests for the PhysioNet 2012 loader.

These tests don't download the real dataset (too large for CI); they
verify the parser against synthetic input files that match the exact
on-disk format of the PhysioNet 2012 challenge files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clinical_llm.data.physionet2012_loader import (
    NAME_REWRITES,
    SENTINELS,
    _parse_one_patient,
    load,
)


def _write_patient_file(path: Path, record_id: int, rows: list[tuple[str, str, float]]) -> None:
    """Write a single patient CSV in PhysioNet 2012 format."""
    lines = ["Time,Parameter,Value"]
    lines.append(f"00:00,RecordID,{record_id}")
    for time, param, value in rows:
        lines.append(f"{time},{param},{value}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def fake_raw_dir(tmp_path: Path) -> Path:
    """Synthesise a tiny PhysioNet-2012-shaped dataset on disk."""
    raw_dir = tmp_path / "raw"
    set_a = raw_dir / "set-a"
    set_b = raw_dir / "set-b"
    set_a.mkdir(parents=True)
    set_b.mkdir(parents=True)

    # Two patients in set-a
    _write_patient_file(
        set_a / "132539.txt",
        record_id=132539,
        rows=[
            ("00:00", "Age", 75),
            ("00:00", "Gender", 0),
            ("00:00", "Height", 170.0),
            ("00:00", "ICUType", 4),
            ("00:00", "Weight", 80.0),
            ("00:15", "HR", 88),
            ("00:15", "Temp", 36.8),
            ("01:00", "HR", 92),
            ("01:00", "Lactate", 2.1),
            ("01:00", "Creatinine", 1.1),
            ("02:00", "HR", -1),  # sentinel
        ],
    )
    _write_patient_file(
        set_a / "132540.txt",
        record_id=132540,
        rows=[
            ("00:00", "Age", 62),
            ("00:00", "Gender", 1),
            ("00:30", "HR", 105),
            ("00:30", "Lactate", 4.0),
        ],
    )
    # One patient in set-b
    _write_patient_file(
        set_b / "142001.txt",
        record_id=142001,
        rows=[
            ("00:00", "Age", 58),
            ("00:00", "Gender", 1),
            ("00:45", "GCS", 13),
            ("01:30", "FiO2", 0.4),
        ],
    )

    # Outcomes files for both sets
    (raw_dir / "Outcomes-a.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n"
        "132539,16,5,5,-1,0\n"
        "132540,21,7,12,8,1\n"
    )
    (raw_dir / "Outcomes-b.txt").write_text(
        "RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death\n" "142001,12,3,3,-1,0\n"
    )
    return raw_dir


def test_parse_one_patient_extracts_static_and_events(fake_raw_dir):
    path = fake_raw_dir / "set-a" / "132539.txt"
    static, events = _parse_one_patient(path)
    assert static["patient_id"] == 132539
    assert static["age"] == 75
    assert static["sex"] == "F"
    # 4 valid events (HR x2, Temp, Lactate, Creatinine) — sentinel HR=-1 dropped.
    assert len(events) == 5
    # Vital names should be rewritten to project convention.
    vital_names = {e["vital_name"] for e in events}
    assert "heart_rate" in vital_names
    assert "lactate" in vital_names


def test_sentinel_values_are_dropped(fake_raw_dir):
    path = fake_raw_dir / "set-a" / "132539.txt"
    _, events = _parse_one_patient(path)
    # No event should carry the sentinel value -1.
    for e in events:
        assert e["value"] not in SENTINELS


def test_gender_maps_to_F_M(fake_raw_dir):
    static_a, _ = _parse_one_patient(fake_raw_dir / "set-a" / "132539.txt")
    static_b, _ = _parse_one_patient(fake_raw_dir / "set-a" / "132540.txt")
    assert static_a["sex"] == "F"  # gender 0
    assert static_b["sex"] == "M"  # gender 1


def test_load_produces_project_schema(fake_raw_dir, tmp_path):
    out_dir = tmp_path / "out"
    patients, events = load(fake_raw_dir, out_dir)

    # patients.csv schema
    assert set(["patient_id", "age", "sex", "in_hospital_mortality"]).issubset(patients.columns)
    # events.csv schema
    assert set(["patient_id", "charttime", "vital_name", "value", "unit"]).issubset(events.columns)
    # CSVs were written to disk
    assert (out_dir / "patients.csv").exists()
    assert (out_dir / "events.csv").exists()


def test_load_attaches_outcomes(fake_raw_dir, tmp_path):
    patients, _ = load(fake_raw_dir, tmp_path / "out")
    outcomes = patients.set_index("patient_id")["in_hospital_mortality"]
    assert outcomes[132539] == 0  # set-a survivor
    assert outcomes[132540] == 1  # set-a non-survivor
    assert outcomes[142001] == 0  # set-b survivor


def test_load_combines_both_sets(fake_raw_dir, tmp_path):
    patients, _ = load(fake_raw_dir, tmp_path / "out")
    assert len(patients) == 3  # 2 from set-a + 1 from set-b


def test_load_pipeline_integrates_with_features(fake_raw_dir, tmp_path):
    """Smoke test: the loader's output should be consumable by the rest of the pipeline."""
    from clinical_llm.data.features import extract_features

    patients, events = load(fake_raw_dir, tmp_path / "out")
    features = extract_features(patients, events)
    # One row per patient.
    assert len(features) == len(patients)
    # patient_id should be preserved.
    assert "patient_id" in features.columns


def test_name_rewrites_are_lowercase():
    """All target names in NAME_REWRITES should be lowercase to match the project convention."""
    for target in NAME_REWRITES.values():
        assert target == target.lower()
