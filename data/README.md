# Data

This directory is intentionally mostly empty in version control. Clinical
data is **never** committed.

## Synthetic data

Generated locally on demand:

```bash
python -m clinical_llm.data.synthetic_generator --n-patients 1000
```

Outputs go to `data/synthetic/` (gitignored).

## MIMIC-IV

Once your PhysioNet credentialing is approved, place MIMIC-IV tables in
`data/mimic_iv/`. The pipeline expects:

- `patients.csv` with columns: `patient_id, age, sex, in_hospital_mortality`
- `events.csv`   with columns: `patient_id, charttime, vital_name, value, unit`

A converter from raw MIMIC-IV tables to this schema is on the roadmap.

## Why no committed data?

MIMIC-IV is credentialed-access and explicitly prohibits redistribution.
Even fully synthetic data is omitted from version control to keep the
repository small and to make it obvious to readers that *real data is not
here*.
