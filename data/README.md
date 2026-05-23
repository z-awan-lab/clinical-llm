# Data

This directory is intentionally mostly empty in version control. Clinical
data is **never** committed.

## PhysioNet 2012 (primary)

Downloaded and converted in two commands:

```bash
python -m clinical_llm.data.physionet2012_downloader
python -m clinical_llm.data.physionet2012_loader
```

Raw files land in `data/physionet2012/raw/`; the converted CSVs in the
project's standard schema (`patients.csv`, `events.csv`) land in
`data/physionet2012/`.

The dataset is licensed under the
[Open Data Commons Attribution Licence](https://physionet.org/about/database/)
and does not require credentialing.

## Synthetic

Generated locally on demand:

```bash
python -m clinical_llm.data.synthetic_generator --n-patients 1000
```

Outputs go to `data/synthetic/` (gitignored). Used for pipeline
verification and CI; not intended for research-grade evaluation.

## MIMIC-IV (optional external validation)

For users with credentialed PhysioNet access. Place MIMIC-IV tables
converted to the project schema in `data/mimic_iv/`:

- `patients.csv` with columns: `patient_id, age, sex, in_hospital_mortality`
- `events.csv` with columns: `patient_id, charttime, vital_name, value, unit`

A dedicated MIMIC-IV converter is on the roadmap.

## Why no committed data?

MIMIC-IV is credentialed-access and explicitly prohibits redistribution.
PhysioNet 2012 is openly licensed but easily redownloadable, so we omit
it from version control to keep the repository small and to make it
obvious that downloaded data is not the source of truth.
