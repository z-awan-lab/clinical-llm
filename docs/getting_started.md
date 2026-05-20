# Getting started

This guide walks you through running the `clinical-llm` pipeline end-to-end —
first on synthetic data (no credentials needed), then on real MIMIC-IV.

## 1. Install

Requirements: Python 3.10 or newer.

```bash
git clone https://github.com/z-awan-lab/clinical-llm.git
cd clinical-llm
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Verify the install:

```bash
pytest tests/ -v
```

You should see 17 tests pass.

## 2. Run on synthetic data

The synthetic generator produces data that mirrors MIMIC-IV's schema, so the
same pipeline runs on either source. Generate a small cohort and train:

```bash
python -m clinical_llm.data.synthetic_generator --n-patients 1000
python -m clinical_llm.training.train --data-dir data/synthetic
```

Outputs are saved to `outputs/baseline_logreg/`:

- `results.json` — point estimates and bootstrap 95% CIs for AUROC, AUPRC, Brier.
- `coefficients.csv` — logistic regression coefficients sorted by absolute magnitude.

## 3. Use real MIMIC-IV

### 3.1 Get credentialed

MIMIC-IV requires PhysioNet credentialing. See the
[official instructions](https://physionet.org/about/credentialing/). Briefly:

1. Create a [PhysioNet](https://physionet.org) account.
2. Complete the **"Data or Specimens Only Research"** CITI course (free) and
   download the *completion report* (not the certificate).
3. Submit a credentialing application listing a senior academic reference.
4. Sign the MIMIC-IV Data Use Agreement once approved.

Approval typically takes 3–14 days.

### 3.2 Download MIMIC-IV

Two routes:

- **Direct download** from
  [physionet.org/content/mimiciv](https://physionet.org/content/mimiciv/).
  Largest tables compressed are around 60GB.
- **Google Cloud BigQuery** — query MIMIC-IV without downloading. Recommended
  for prototyping. See [PhysioNet's BigQuery guide](https://mimic.mit.edu/docs/gettingstarted/cloud/bigquery/).

### 3.3 Convert MIMIC-IV to the expected schema

The pipeline expects two CSVs:

- `patients.csv` with columns: `patient_id, age, sex, in_hospital_mortality`
- `events.csv`   with columns: `patient_id, charttime, vital_name, value, unit`

A converter script (`src/clinical_llm/data/mimic_loader.py`) is on the roadmap.
For now, the synthetic schema documents the expected shape exactly.

## 4. What's next

Once the logistic regression baseline runs, the project roadmap is:

- [x] Logistic regression baseline
- [ ] XGBoost baseline
- [ ] LSTM baseline on raw sequences
- [ ] Llama-3.2-3B + LoRA fine-tuning
- [ ] SHAP interpretability for the baselines
- [ ] Attention visualisation for the LLM
- [ ] Streamlit demo

See [`docs/design.md`](design.md) for design rationale.
