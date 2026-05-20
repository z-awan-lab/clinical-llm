"""Feature extraction for clinical time-series.

Converts the long-form events table into a flat feature matrix suitable for
classical baselines (logistic regression, XGBoost). For each patient and each
vital sign, we compute summary statistics over the observation window.

This is intentionally simple — the point of having the LLM is to do better
than this. If the LLM doesn't beat aggregated features by a meaningful
margin, that's a real finding worth reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .synthetic_generator import VITALS

# Summary statistics computed per vital per patient.
# Each becomes a column in the feature matrix.
SUMMARY_STATS = ["mean", "min", "max", "std", "first", "last", "count"]


def _summarise(series: pd.Series) -> dict[str, float]:
    """Compute summary statistics for a single vital's measurements."""
    values = series.dropna().values
    if len(values) == 0:
        # No measurements — return NaN for all stats; imputation handled later.
        return {stat: np.nan for stat in SUMMARY_STATS}
    return {
        "mean":  float(np.mean(values)),
        "min":   float(np.min(values)),
        "max":   float(np.max(values)),
        "std":   float(np.std(values)) if len(values) > 1 else 0.0,
        "first": float(values[0]),
        "last":  float(values[-1]),
        "count": float(len(values)),
    }


def extract_features(
    patients: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build a flat feature matrix: one row per patient.

    Args:
        patients: static patient features (must include patient_id, age, sex).
        events:   long-form events (patient_id, charttime, vital_name, value).

    Returns:
        DataFrame with one row per patient. Columns include static features
        and summary statistics for each vital. Missing measurements are
        imputed with the global median during model fitting, not here.
    """
    # Sort events by patient and time so 'first' and 'last' are well-defined.
    events = events.sort_values(["patient_id", "charttime"])

    # Wide-format feature dict per patient.
    feature_rows: list[dict] = []
    grouped = events.groupby(["patient_id", "vital_name"])

    # Precompute summaries: {(pid, vital): {stat: value}}
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for (pid, vital_name), group in grouped:
        summaries[(pid, vital_name)] = _summarise(group["value"])

    for _, patient in patients.iterrows():
        pid = patient["patient_id"]
        row: dict[str, float | int | str] = {
            "patient_id": pid,
            "age": patient["age"],
            "sex_F": int(patient["sex"] == "F"),
        }
        for vital_name in VITALS:
            stats = summaries.get((pid, vital_name), {s: np.nan for s in SUMMARY_STATS})
            for stat_name, value in stats.items():
                row[f"{vital_name}_{stat_name}"] = value
        feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def get_label_column() -> str:
    """Name of the binary outcome column in the patients table."""
    return "in_hospital_mortality"


def feature_columns(features: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes patient_id)."""
    return [c for c in features.columns if c != "patient_id"]
