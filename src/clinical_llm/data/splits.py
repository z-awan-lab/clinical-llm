"""Patient-level train/val/test splits.

Critical: splits MUST be at the patient level, never at the event level.
A patient's events appearing in both train and test would leak information
and inflate metrics. This is a common mistake in clinical ML.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    patients: pd.DataFrame,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_on: str = "in_hospital_mortality",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split patients into train, val, test with stratification on the outcome.

    Returns:
        (train_patients, val_patients, test_patients) — each is a patients DataFrame.
    """
    # First split: train+val vs test
    trainval, test = train_test_split(
        patients,
        test_size=test_size,
        stratify=patients[stratify_on],
        random_state=seed,
    )
    # Second split: train vs val (val_size is relative to trainval)
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(
        trainval,
        test_size=val_relative,
        stratify=trainval[stratify_on],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def filter_events_by_patients(
    events: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only events for the given patient cohort."""
    return events[events["patient_id"].isin(patients["patient_id"])].copy()


def describe_split(name: str, patients: pd.DataFrame) -> None:
    """Print a one-line summary of a split — useful for logs."""
    n = len(patients)
    mortality = patients["in_hospital_mortality"].mean()
    print(f"  {name:>5}: {n:>5,} patients, {mortality:.1%} mortality")
