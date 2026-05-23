# Getting started

This guide walks you through running the `clinical-llm` pipeline end-to-end —
first on synthetic data, then on the public PhysioNet/CinC Challenge 2012
dataset, with an optional MIMIC-IV path for users who hold credentialed
access.

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

You should see all tests pass (some LLM tests are skipped unless
transformers and peft are installed via `pip install -e ".[llm]"`).

## 2. Run on synthetic data

The synthetic generator produces data in the project's standard schema,
so the same pipeline runs on either source. Generate a small cohort
and train:

```bash
python -m clinical_llm.data.synthetic_generator --n-patients 1000
python -m clinical_llm.training.train --model logreg --data-dir data/synthetic
```

Outputs are saved to `outputs/baseline_logreg/`:

- `results.json` — point estimates and bootstrap 95% CIs for AUROC, AUPRC, Brier.
- `coefficients.csv` — logistic regression coefficients sorted by absolute magnitude.

Synthetic-data metrics are deliberately easy and should not be read as
research results — they verify the pipeline runs end-to-end.

## 3. Run on real ICU data (PhysioNet 2012)

The PhysioNet/CinC Challenge 2012 dataset is publicly downloadable with
no credentialing or DUA. The full release contained 12,000 ICU patients
across three subsets, but only **Set A** (approximately 4,000 patients)
has publicly released outcome labels — Sets B and C were withheld by
PhysioNet for evaluation purposes. This project uses Set A.

```bash
# Download (~3MB compressed) and extract Set A
python -m clinical_llm.data.physionet2012_downloader

# Convert to the project's standard patients.csv / events.csv schema
python -m clinical_llm.data.physionet2012_loader

# Train any model
python -m clinical_llm.training.train --model logreg --data-dir data/physionet2012
python -m clinical_llm.training.train --model xgboost --data-dir data/physionet2012
python -m clinical_llm.training.train --model lstm --data-dir data/physionet2012

# MedGemma 4B + LoRA — requires a GPU and Hugging Face gated access
pip install -e ".[llm]"
huggingface-cli login   # accept Gemma terms at https://huggingface.co/google/medgemma-4b-it
python -m clinical_llm.training.train --model llm --data-dir data/physionet2012
```

Cite the dataset as:

> Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality
> of ICU patients: The PhysioNet/Computing in Cardiology Challenge 2012.
> Computing in Cardiology 2012; 39: 245-248.

## 4. Optional: MIMIC-IV external validation

For users with [credentialed PhysioNet access](https://physionet.org/about/credentialing/),
MIMIC-IV is supported as an external validation cohort. The pipeline
expects MIMIC-IV converted to the project's standard schema:

- `patients.csv` with columns: `patient_id, age, sex, in_hospital_mortality`
- `events.csv` with columns: `patient_id, charttime, vital_name, value, unit`

A dedicated MIMIC-IV loader is on the roadmap; the schema mirrors the
PhysioNet 2012 loader's output exactly, so the conversion is mechanical.

## 5. Project roadmap

- [x] Logistic regression baseline
- [x] XGBoost baseline
- [x] LSTM baseline on raw sequences
- [x] MedGemma 4B + LoRA fine-tuning
- [x] PhysioNet 2012 loader
- [ ] SHAP interpretability for baselines
- [ ] Attention visualisation for the LLM
- [ ] Streamlit demo
- [ ] MIMIC-IV loader for external validation

See [`docs/design.md`](design.md) for the rationale behind each choice.
