# clinical-llm

> Fine-tuning open LLMs on clinical sequences for outcome prediction. MIMIC-IV benchmarks, baselines, and interpretability.

[![Tests](https://github.com/z-awan-lab/clinical-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/z-awan-lab/clinical-llm/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An open-source pipeline for fine-tuning small open-weight language models on longitudinal electronic health record (EHR) data, benchmarked on the MIMIC-IV in-hospital mortality prediction task.

## Motivation

Large language models have shown promise for clinical prediction tasks, but most published work depends on private models, private data, or both. This repository provides a fully reproducible, open-stack alternative:

- **Open models** — Llama-3.2-3B and Phi-3-mini via Hugging Face
- **Open data** — MIMIC-IV (credentialed but public), with a synthetic data fallback so the pipeline runs without credentials
- **Open evaluation** — standard MIMIC benchmark splits, with bootstrap confidence intervals and calibration analysis
- **Honest baselines** — logistic regression, XGBoost, and LSTM, so the LLM has to earn its complexity

## Results (work in progress)

| Model                          | AUROC          | AUPRC          | Brier ↓        |
| ------------------------------ | -------------- | -------------- | -------------- |
| Logistic Regression            | _coming soon_  | _coming soon_  | _coming soon_  |
| XGBoost                        | _coming soon_  | _coming soon_  | _coming soon_  |
| LSTM (raw sequences)           | _coming soon_  | _coming soon_  | _coming soon_  |
| Llama-3.2-3B + LoRA            | _coming soon_  | _coming soon_  | _coming soon_  |

Results table will be updated as experiments complete. Bootstrap 95% CIs reported in [`docs/results.md`](docs/results.md).

## Quick start

```bash
# Clone and install
git clone https://github.com/z-awan-lab/clinical-llm.git
cd clinical-llm
pip install -e ".[dev]"

# Run the full pipeline on synthetic data (no credentials needed)
python -m clinical_llm.training.train --config configs/baseline_logreg.yaml

# Evaluate
python -m clinical_llm.evaluation.evaluate --run-dir outputs/baseline_logreg
```

See [`docs/getting_started.md`](docs/getting_started.md) for the full walkthrough including MIMIC-IV setup.

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

**In-hospital mortality prediction** using the first 48 hours of ICU data. This is the canonical MIMIC benchmark task, defined identically to [Harutyunyan et al., 2019](https://www.nature.com/articles/s41597-019-0103-9), allowing direct comparison with published results.

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

Built on the shoulders of [MIMIC-IV](https://physionet.org/content/mimiciv/), [Hugging Face Transformers](https://github.com/huggingface/transformers), and the [PEFT](https://github.com/huggingface/peft) library.
