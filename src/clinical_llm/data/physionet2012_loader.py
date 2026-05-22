"""Convert PhysioNet/CinC Challenge 2012 raw files into the project schema.

This loader reads the per-patient CSV files from sets A and B together with
the corresponding outcome files, and emits two tables matching the same
schema used by the synthetic generator:

    patients.csv : patient_id, age, sex, in_hospital_mortality
    events.csv   : patient_id, charttime, vital_name, value, unit

After this conversion, every downstream module (features, sequences,
serialisation, training) consumes the data through the same interface —
no model code changes are needed.

The 2012 dataset has up to 41 variables per patient. We pass through all
of the time-series variables in the dataset and report the names that
were actually observed in the cohort. Static descriptors (age, sex,
height, weight, ICU type) live on the patient row; everything else
becomes time-stamped events.

Usage:
    python -m clinical_llm.data.physionet2012_loader \\
        --raw-dir data/physionet2012/raw \\
        --out-dir data/physionet2012
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Variables in the 2012 dataset that we treat as static descriptors,
# stored on the patient row rather than as time-series events.
STATIC_DESCRIPTORS = {"Age", "Gender", "Height", "Weight", "ICUType", "RecordID"}

# Sentinel values used by the 2012 challenge for missing data.
SENTINELS = {-1.0, -1}

# Map the dataset's variable names to lowercased names consistent with the
# rest of the project. Anything not in this map flows through with its
# original casing — keeps the loader future-proof if PhysioNet ever
# adds new variables.
NAME_REWRITES = {
    "HR": "heart_rate",
    "NIDiasABP": "dbp_noninvasive",
    "NISysABP": "sbp_noninvasive",
    "DiasABP": "dbp",
    "SysABP": "sbp",
    "MAP": "map",
    "NIMAP": "map_noninvasive",
    "RespRate": "respiratory_rate",
    "SaO2": "spo2",
    "Temp": "temperature",
    "Glucose": "glucose",
    "Lactate": "lactate",
    "Creatinine": "creatinine",
    "BUN": "bun",
    "Bilirubin": "bilirubin",
    "WBC": "wbc",
    "Platelets": "platelets",
    "HCO3": "hco3",
    "FiO2": "fio2",
    "GCS": "gcs",
    "Urine": "urine_output",
    "MechVent": "mechanical_ventilation",
}


def _parse_time(s: str) -> pd.Timedelta:
    """The 2012 files use 'HH:MM' elapsed time since ICU admission."""
    h, m = s.strip().split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m))


def _read_outcomes(raw_dir: Path) -> pd.DataFrame:
    """Read Outcomes-a.txt and Outcomes-b.txt and concatenate."""
    frames = []
    for name in ("Outcomes-a.txt", "Outcomes-b.txt"):
        path = raw_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run physionet2012_downloader first.")
        df = pd.read_csv(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _patient_files(raw_dir: Path) -> list[Path]:
    """Find all per-patient CSV files across set-a and set-b directories."""
    files: list[Path] = []
    for subset in ("set-a", "set-b"):
        subset_dir = raw_dir / subset
        if not subset_dir.exists():
            raise FileNotFoundError(f"{subset_dir} not found. Did the tarball extract correctly?")
        files.extend(sorted(subset_dir.glob("*.txt")))
    return files


def _parse_one_patient(
    path: Path,
) -> tuple[dict, list[dict]]:
    """Parse one patient's file into a static-record dict and an events list."""
    df = pd.read_csv(path)
    # Schema: Time, Parameter, Value
    df = df.dropna(subset=["Parameter", "Value"])

    # Static descriptors are encoded as Time=00:00 with Parameter in STATIC_DESCRIPTORS.
    static_mask = df["Parameter"].isin(STATIC_DESCRIPTORS)
    static_rows = df[static_mask]
    static = {row["Parameter"]: row["Value"] for _, row in static_rows.iterrows()}

    pid = int(static.get("RecordID", path.stem))

    # Time-series events: everything else.
    ts_df = df[~static_mask].copy()
    # Drop sentinel missing values.
    ts_df = ts_df[~ts_df["Value"].isin(SENTINELS)]

    # 2012 data has elapsed-time stamps; anchor to a fixed dummy admission
    # date so downstream code (which expects datetime) works without
    # special-casing this dataset.
    admission = pd.Timestamp("2000-01-01")
    events: list[dict] = []
    for _, row in ts_df.iterrows():
        try:
            elapsed = _parse_time(str(row["Time"]))
        except (ValueError, AttributeError):
            continue
        var = str(row["Parameter"])
        events.append(
            {
                "patient_id": pid,
                "charttime": admission + elapsed,
                "vital_name": NAME_REWRITES.get(var, var.lower()),
                "value": float(row["Value"]),
                "unit": "",  # 2012 dataset does not provide unit annotations
            }
        )

    age = float(static.get("Age", float("nan")))
    gender_code = static.get("Gender", -1)
    sex = "F" if gender_code == 0 else "M" if gender_code == 1 else "U"
    return (
        {"patient_id": pid, "age": age, "sex": sex},
        events,
    )


def load(raw_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert raw PhysioNet 2012 files to patients.csv and events.csv.

    Returns the two DataFrames and writes them to ``out_dir``.
    """
    outcomes = _read_outcomes(raw_dir)
    outcomes = outcomes.set_index("RecordID")
    if "In-hospital_death" not in outcomes.columns:
        raise ValueError("Outcomes file missing expected column 'In-hospital_death'.")

    files = _patient_files(raw_dir)
    print(f"Parsing {len(files):,} patient files...")

    patient_rows: list[dict] = []
    all_events: list[dict] = []

    for i, path in enumerate(files, 1):
        if i % 500 == 0:
            print(f"  {i:,}/{len(files):,}")
        static, events = _parse_one_patient(path)
        pid = static["patient_id"]
        if pid not in outcomes.index:
            # Should not happen for sets a+b, but skip defensively.
            continue
        static["in_hospital_mortality"] = int(outcomes.loc[pid, "In-hospital_death"])
        patient_rows.append(static)
        all_events.extend(events)

    patients_df = pd.DataFrame(patient_rows)
    events_df = pd.DataFrame(all_events)

    # Drop patients with no events or invalid age.
    patients_df = patients_df.dropna(subset=["age"])
    patients_df = patients_df[patients_df["age"] > 0]
    valid_pids = set(patients_df["patient_id"])
    events_df = events_df[events_df["patient_id"].isin(valid_pids)]

    out_dir.mkdir(parents=True, exist_ok=True)
    patients_path = out_dir / "patients.csv"
    events_path = out_dir / "events.csv"
    patients_df.to_csv(patients_path, index=False)
    events_df.to_csv(events_path, index=False)

    n_pos = int(patients_df["in_hospital_mortality"].sum())
    n_vars = events_df["vital_name"].nunique()
    print()
    print(
        f"Patients : {len(patients_df):,} ({n_pos:,} positive, "
        f"{n_pos / len(patients_df):.1%} mortality)"
    )
    print(f"Events   : {len(events_df):,}")
    print(f"Variables: {n_vars} unique time-series parameters")
    print(f"Written  : {patients_path}, {events_path}")

    return patients_df, events_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/physionet2012/raw"),
        help="Directory containing extracted set-a/, set-b/, and outcome files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/physionet2012"),
        help="Destination directory for patients.csv and events.csv.",
    )
    args = parser.parse_args()
    load(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
