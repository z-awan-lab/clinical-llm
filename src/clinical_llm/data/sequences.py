"""Sequence dataset for sequence-based models.

Converts the long-form events table into per-patient time-aligned
sequences suitable for an LSTM or other sequence model. Unlike the flat
feature extraction used by logistic regression and XGBoost, this
preserves the temporal ordering and per-timestep granularity.

Each patient becomes a tensor of shape (T, F) where T is the number of
timesteps (1 per hour over the observation window) and F is the number
of vital signs. Missing measurements are represented with an explicit
mask channel rather than zero-imputed, so the model can learn that
"absent" is itself informative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .synthetic_generator import VITALS


@dataclass
class SequenceConfig:
    """Configuration for building patient sequences."""

    observation_hours: int = 48
    bucket_hours: int = 1  # one bucket per hour


def _build_pivot(events: pd.DataFrame, config: SequenceConfig) -> dict[int, pd.DataFrame]:
    """Pivot events into a per-patient (timestep × vital) matrix.

    Returns a dict {patient_id: DataFrame of shape (T, n_vitals)}, where
    missing values remain NaN.
    """
    events = events.copy()
    events["charttime"] = pd.to_datetime(events["charttime"])

    # Normalise each patient's timeline to start at zero, then bucket by hour.
    events = events.sort_values(["patient_id", "charttime"])
    events["t0"] = events.groupby("patient_id")["charttime"].transform("min")
    events["hour"] = ((events["charttime"] - events["t0"]).dt.total_seconds() // 3600).astype(int)
    events = events[events["hour"] < config.observation_hours]

    vital_order = list(VITALS.keys())
    n_t = config.observation_hours

    out: dict[int, pd.DataFrame] = {}
    for pid, group in events.groupby("patient_id"):
        # Average if multiple measurements fall in the same hourly bucket.
        wide = group.pivot_table(
            index="hour",
            columns="vital_name",
            values="value",
            aggfunc="mean",
        ).reindex(index=range(n_t), columns=vital_order)
        out[int(pid)] = wide
    return out


def _normalise(matrix: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """Z-score using precomputed train-set mean and std."""
    return (matrix - stats["mean"]) / np.where(stats["std"] > 0, stats["std"], 1.0)


def compute_normalisation_stats(
    pivot: dict[int, pd.DataFrame],
) -> dict[str, np.ndarray]:
    """Compute per-vital mean and std across all training observations.

    Skips NaN values. Returned arrays have shape (n_vitals,).
    """
    stacked = np.concatenate([df.values for df in pivot.values()], axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    # Replace any NaN means (no observations of a vital at all) with 0.
    mean = np.where(np.isnan(mean), 0.0, mean)
    std = np.where(np.isnan(std), 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


class SequenceDataset(Dataset):
    """PyTorch Dataset producing (values, mask, label) per patient.

    The mask channel is essential: it lets the model distinguish "the
    value was zero" from "the value is missing." Real clinical data is
    sparse, and how the model handles that sparsity is a recurring
    research question.
    """

    def __init__(
        self,
        patients: pd.DataFrame,
        events: pd.DataFrame,
        stats: dict[str, np.ndarray],
        config: SequenceConfig | None = None,
        label_column: str = "in_hospital_mortality",
    ) -> None:
        self.config = config or SequenceConfig()
        self.stats = stats

        pivot = _build_pivot(events, self.config)
        patient_order = patients["patient_id"].tolist()

        self.values: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []
        self.labels: list[int] = []

        n_t = self.config.observation_hours
        n_f = len(VITALS)

        labels_by_pid = patients.set_index("patient_id")[label_column].to_dict()

        for pid in patient_order:
            df = pivot.get(int(pid))
            if df is None:
                values = np.full((n_t, n_f), np.nan, dtype=np.float32)
            else:
                values = df.values.astype(np.float32)

            mask = (~np.isnan(values)).astype(np.float32)
            values = np.where(mask.astype(bool), values, 0.0)
            values = _normalise(values, stats).astype(np.float32)
            # Re-zero any positions that were missing (normalisation may have shifted them).
            values = values * mask

            self.values.append(values)
            self.masks.append(mask)
            self.labels.append(int(labels_by_pid[pid]))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = torch.from_numpy(self.values[idx])
        mask = torch.from_numpy(self.masks[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return values, mask, label


def n_features() -> int:
    """Number of vital signs in the sequence representation."""
    return len(VITALS)
