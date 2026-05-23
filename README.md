# clinical-llm

> Fine-tuning open LLMs on clinical sequences for outcome prediction. Public ICU benchmarks, honest baselines, and interpretability.

[![Tests](https://github.com/z-awan-lab/clinical-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/z-awan-lab/clinical-llm/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An open-source pipeline for fine-tuning small open-weight language models on
longitudinal electronic health record (EHR) data, benchmarked on in-hospital
mortality prediction from the first 48 hours of ICU stay.

## Motivation

Large language models have shown promise for clinical prediction tasks, but most published work depends on private models, private data, or both. This repository provides a fully reproducible, open-stack alternative:

- **Open models** — MedGemma 4B (medical-pretrained Gemma family) via Hugging Face, with LoRA parameter-efficient fine-tuning. The model identifier is configurable, so general-purpose alternatives such as Llama-3.2-3B can be swapped in.
- **Open data** — Primary benchmarks run on the [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/) dataset (~4,000 publicly labelled ICU patients from Set A, no credentialing required). MIMIC-IV is supported as an optional external validation cohort. A synthetic data generator is also included so the pipeline runs end-to-end with neither.
- **Open evaluation** — patient-level splits, bootstrap 95% confidence intervals on AUROC / AUPRC / Brier, calibration analysis
- **Honest baselines** — logistic regression, XGBoost, and LSTM, so the LLM has to earn its complexity

## Results

Benchmarks are run on the **PhysioNet/CinC Challenge 2012** Set A
(approximately 4,000 ICU patients with publicly released outcome labels).
Only Set A has publicly released outcomes — Sets B and C were withheld by
PhysioNet for evaluation purposes. MIMIC-IV is supported as an optional
external validation cohort for users with credentialed access.

| Model                          | Status          | PhysioNet 2012 AUROC | PhysioNet 2012 AUPRC |
| ------------------------------ | --------------- | -------------------- | -------------------- |
| Logistic Regression            | ✅ implemented  | _running_            | _running_            |
| XGBoost                        | ✅ implemented  | _running_            | _running_            |
| LSTM (raw sequences)           | ✅ implemented  | _running_            | _running_            |
| MedGemma 4B + LoRA             | ✅ implemented  | _running_            | _running_            |

Bootstrap 95% CIs and calibration plots are reported in [`docs/results.md`](docs/results.md).

## Quick start

```bash
# Clone and install
git clone https://github.com/z-awan-lab/clinical-llm.git
cd clinical-llm
pip install -e ".[dev]"

# OPTION 1 — quick start on synthetic data (no downloads needed)
python -m clinical_llm.data.synthetic_generator --n-patients 2000
python -m clinical_llm.training.train --model logreg --data-dir data/synthetic
python -m clinical_llm.training.train --model xgboost --data-dir data/synthetic
python -m clinical_llm.training.train --model lstm --data-dir data/synthetic

# OPTION 2 — real ICU data (PhysioNet 2012, publicly downloadable)
python -m clinical_llm.data.physionet2012_downloader
python -m clinical_llm.data.physionet2012_loader
python -m clinical_llm.training.train --model logreg --data-dir data/physionet2012
python -m clinical_llm.training.train --model xgboost --data-dir data/physionet2012
python -m clinical_llm.training.train --model lstm --data-dir data/physionet2012

# MedGemma 4B + LoRA (requires GPU and Hugging Face gated access)
pip install -e ".[llm]"
huggingface-cli login   # accept Gemma terms at https://huggingface.co/google/medgemma-4b-it
python -m clinical_llm.training.train --model llm --data-dir data/physionet2012
```

See [`docs/getting_started.md`](docs/getting_started.md) for the full walkthrough, including the optional MIMIC-IV external validation path.

## Project structure

```
clinical-llm/
├── src/clinical_llm/
│   ├── data/              # Loaders, tokenisation, train/val/test splits
│   ├── models/            # Baselines + LoRA fine-tuning
│   ├── training/          # Training loops and configs
│   ├── evaluation/        # Metrics, calibration, bootstrap CIs
│   └── interpretability/  # SHAP and attention visualisation
├── configs/               # YAML configs for each experiment
├── notebooks/             # Exploration and results visualisation
├── tests/                 # pytest unit tests
├── app/                   # Streamlit demo
└── docs/                  # Design decisions and results
```

## Task definition

**In-hospital mortality prediction** from the first 48 hours of ICU stay.
This is a canonical clinical ML benchmark; the prediction setup follows
the convention used in [Harutyunyan et al., 2019](https://www.nature.com/articles/s41597-019-0103-9),
allowing results to be compared with the broader literature on MIMIC,
PhysioNet 2012, and eICU benchmarks.

## Reproducibility

- All experiments are deterministic given a fixed seed (`--seed 42` by default).
- Pinned dependencies in `pyproject.toml`.
- Dockerfile included for environment isolation.
- CI runs the full test suite on every push.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@software{awan2026clinicalllm,
  author = {Awan, Zainab},
  title  = {clinical-llm: Fine-tuning open LLMs on clinical sequences},
  year   = {2026},
  url    = {https://github.com/z-awan-lab/clinical-llm}
}
```

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built on the shoulders of the [PhysioNet/CinC Challenge 2012](https://physionet.org/content/challenge-2012/) (Silva et al., 2012), [MIMIC-IV](https://physionet.org/content/mimiciv/), [Hugging Face Transformers](https://github.com/huggingface/transformers), and the [PEFT](https://github.com/huggingface/peft) library.
