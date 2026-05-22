"""Serialise clinical sequences into text prompts for LLM consumption.

Design choice: we represent each patient as a structured text prompt of
the form:

    Patient: 67-year-old F
    Hour 0: HR 88, SBP 124, ... (only observed vitals)
    Hour 1: HR 92, ...
    ...
    Hour 47: ...
    Outcome:

We chose this representation over the alternatives for these reasons:

  * Compared to a raw numeric matrix, text plays to the LLM's strengths —
    it has seen millions of clinical notes during pretraining, including
    via MedGemma's medical pretraining corpus.
  * Compared to free-form notes ("the patient was tachycardic at hour 2..."),
    structured rows preserve temporal alignment unambiguously and avoid
    asking the LLM to do unnecessary parsing work.
  * Compared to one-shot summarisation ("vitals over 48 hours: HR mean 90..."),
    the per-timestep format gives the model the same granularity that the
    LSTM saw — making the comparison between models fair.

We deliberately omit values that were not observed at a given timestep,
rather than filling with "missing." This both shortens prompts and lets
the model learn that absence is itself a signal — the same principle as
the LSTM's explicit mask channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Short codes keep prompts compact. Token budget matters: at ~6 tokens per
# row × 48 hours × 9 vitals, we're already in the thousands per patient.
VITAL_CODES = {
    "heart_rate": "HR",
    "sbp": "SBP",
    "dbp": "DBP",
    "respiratory_rate": "RR",
    "spo2": "SpO2",
    "temperature": "Temp",
    "glucose": "Gluc",
    "lactate": "Lact",
    "creatinine": "Cr",
}


@dataclass
class SerialisationConfig:
    """Configuration for sequence-to-text serialisation."""

    observation_hours: int = 48
    bucket_hours: int = 1
    include_age_sex: bool = True
    # We always trim to this many tokens during tokenisation downstream;
    # the prompt itself can be longer than this and get truncated.


def _format_value(vital: str, value: float) -> str:
    """Format a single vital sign measurement for a row."""
    code = VITAL_CODES[vital]
    # Match the precision a clinician would actually record.
    if vital == "lactate" or vital == "creatinine":
        return f"{code} {value:.1f}"
    return f"{code} {round(value)}"


def serialise_patient(
    patient_row: pd.Series,
    events: pd.DataFrame,
    config: SerialisationConfig,
) -> str:
    """Serialise one patient's static features and event sequence to text.

    Args:
        patient_row: row from the patients table for this patient. Must
            contain at minimum: patient_id, age, sex.
        events: long-form events filtered to this single patient.
        config: serialisation parameters.

    Returns:
        A single string prompt ending with "Outcome:".
    """
    lines: list[str] = []

    if config.include_age_sex:
        lines.append(f"Patient: {int(round(patient_row['age']))}-year-old {patient_row['sex']}")

    if len(events) == 0:
        lines.append("(no vital sign measurements recorded)")
        lines.append("Outcome:")
        return "\n".join(lines)

    events = events.copy()
    events["charttime"] = pd.to_datetime(events["charttime"])
    t0 = events["charttime"].min()
    events["hour"] = ((events["charttime"] - t0).dt.total_seconds() // 3600).astype(int)
    events = events[events["hour"] < config.observation_hours]

    # Aggregate to one row per (hour, vital) — mean if multiple values fall
    # in the same bucket. This matches the LSTM's bucketing exactly so the
    # two models are comparing like-with-like.
    bucketed = (
        events.groupby(["hour", "vital_name"])["value"]
        .mean()
        .reset_index()
        .sort_values(["hour", "vital_name"])
    )

    for hour, group in bucketed.groupby("hour"):
        parts = [
            _format_value(v, val)
            for v, val in zip(group["vital_name"], group["value"], strict=True)
            if v in VITAL_CODES
        ]
        if parts:
            lines.append(f"Hour {int(hour)}: {', '.join(parts)}")

    lines.append("Outcome:")
    return "\n".join(lines)


def build_prompts(
    patients: pd.DataFrame,
    events: pd.DataFrame,
    config: SerialisationConfig | None = None,
) -> list[str]:
    """Serialise an entire cohort.

    Returns one prompt string per patient, in the order rows appear in
    ``patients``. The labels are *not* attached — the training code adds
    them separately so the same prompt can be reused at inference time
    without the label leaking in.
    """
    config = config or SerialisationConfig()
    events_by_pid = {pid: g for pid, g in events.groupby("patient_id")}

    prompts = []
    for _, row in patients.iterrows():
        pid = int(row["patient_id"])
        patient_events = events_by_pid.get(pid, pd.DataFrame(columns=events.columns))
        prompts.append(serialise_patient(row, patient_events, config))
    return prompts
