"""Synthetic clinical time-series generator.

Generates data that mirrors the MIMIC-IV ICU schema (chartevents-style),
so the entire pipeline can be developed and tested without MIMIC credentials.

The generated data is *intentionally* simplified — it captures the structure
(patient → ICU stays → time-stamped events with measurements) but not the
clinical complexity. Real MIMIC-IV usage swaps this out for the actual loader.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Vital signs and their plausible ranges (loosely based on adult ICU norms).
# These are NOT clinically validated — they exist to give the pipeline
# realistically-shaped numeric data to chew on.
VITALS = {
    "heart_rate": {"healthy": (70, 15), "deteriorating": (110, 20), "unit": "bpm"},
    "sbp": {"healthy": (120, 12), "deteriorating": (95, 15), "unit": "mmHg"},
    "dbp": {"healthy": (75, 8), "deteriorating": (55, 10), "unit": "mmHg"},
    "respiratory_rate": {"healthy": (16, 3), "deteriorating": (24, 5), "unit": "breaths/min"},
    "spo2": {"healthy": (97, 2), "deteriorating": (90, 4), "unit": "%"},
    "temperature": {"healthy": (36.8, 0.4), "deteriorating": (38.5, 0.8), "unit": "C"},
    "glucose": {"healthy": (110, 20), "deteriorating": (180, 50), "unit": "mg/dL"},
    "lactate": {"healthy": (1.2, 0.3), "deteriorating": (3.5, 1.2), "unit": "mmol/L"},
    "creatinine": {"healthy": (0.9, 0.2), "deteriorating": (2.0, 0.7), "unit": "mg/dL"},
}


@dataclass
class GeneratorConfig:
    n_patients: int = 1000
    observation_hours: int = 48
    sampling_interval_hours: int = 1  # one measurement per hour, on average
    mortality_rate: float = 0.10
    missingness_rate: float = 0.15
    seed: int = 42


def _sample_age(rng: np.random.Generator, died: bool) -> float:
    """Patients who die tend to be older on average."""
    mean = 72 if died else 62
    age = rng.normal(loc=mean, scale=14)
    return float(np.clip(age, 18, 95))


def _generate_vital_trajectory(
    rng: np.random.Generator,
    vital_name: str,
    n_timesteps: int,
    deterioration_fraction: float,
) -> np.ndarray:
    """Generate a trajectory that drifts from healthy toward deteriorating.

    deterioration_fraction in [0, 1]: 0 = fully healthy, 1 = fully deteriorating.
    """
    healthy_mean, healthy_sd = VITALS[vital_name]["healthy"]
    sick_mean, sick_sd = VITALS[vital_name]["deteriorating"]

    # Linear interpolation of mean and SD across the stay, with noise.
    fractions = np.linspace(0, deterioration_fraction, n_timesteps)
    means = healthy_mean + fractions * (sick_mean - healthy_mean)
    sds = healthy_sd + fractions * (sick_sd - healthy_sd)
    return rng.normal(loc=means, scale=sds)


def _generate_one_patient(
    rng: np.random.Generator,
    patient_id: int,
    config: GeneratorConfig,
) -> tuple[dict, pd.DataFrame]:
    """Generate one patient: a static record and a long-form events table."""
    died = bool(rng.random() < config.mortality_rate)
    age = _sample_age(rng, died)
    sex = rng.choice(["F", "M"])

    # Patients who die show stronger deterioration in their trajectories.
    # We add per-patient noise so individuals vary.
    base = 0.85 if died else 0.20
    deterioration = float(np.clip(base + rng.normal(0, 0.10), 0, 1))

    n_timesteps = config.observation_hours // config.sampling_interval_hours
    timestamps = pd.date_range(
        start="2150-01-01 00:00",  # MIMIC-IV uses shifted dates; matches that style.
        periods=n_timesteps,
        freq=f"{config.sampling_interval_hours}h",
    )

    rows = []
    for vital_name in VITALS:
        values = _generate_vital_trajectory(rng, vital_name, n_timesteps, deterioration)
        # Introduce missingness — real clinical data is sparse.
        mask = rng.random(n_timesteps) < config.missingness_rate
        values = np.where(mask, np.nan, values)
        for ts, val in zip(timestamps, values, strict=False):
            if np.isnan(val):
                continue
            rows.append(
                {
                    "patient_id": patient_id,
                    "charttime": ts,
                    "vital_name": vital_name,
                    "value": round(float(val), 2),
                    "unit": VITALS[vital_name]["unit"],
                }
            )

    events = pd.DataFrame(rows)

    static = {
        "patient_id": patient_id,
        "age": round(age, 1),
        "sex": sex,
        "in_hospital_mortality": int(died),
    }

    return static, events


def generate(config: GeneratorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic patients matching MIMIC-IV's structure.

    Returns:
        patients: one row per patient with static features and the outcome label.
        events:   long-form table of timestamped measurements.
    """
    rng = np.random.default_rng(config.seed)

    statics, all_events = [], []
    for pid in range(config.n_patients):
        static, events = _generate_one_patient(rng, pid, config)
        statics.append(static)
        all_events.append(events)

    patients = pd.DataFrame(statics)
    events_df = pd.concat(all_events, ignore_index=True)
    return patients, events_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=1000)
    parser.add_argument("--observation-hours", type=int, default=48)
    parser.add_argument("--mortality-rate", type=float, default=0.10)
    parser.add_argument("--missingness-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    config = GeneratorConfig(
        n_patients=args.n_patients,
        observation_hours=args.observation_hours,
        mortality_rate=args.mortality_rate,
        missingness_rate=args.missingness_rate,
        seed=args.seed,
    )

    print(f"Generating {config.n_patients} synthetic patients...")
    patients, events = generate(config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(args.out_dir / "patients.csv", index=False)
    events.to_csv(args.out_dir / "events.csv", index=False)

    print(f"Patients: {len(patients):,} ({patients['in_hospital_mortality'].mean():.1%} mortality)")
    print(f"Events:   {len(events):,}")
    print(f"Written to {args.out_dir}/")


if __name__ == "__main__":
    main()
