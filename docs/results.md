# Results

All metrics report bootstrap 95% confidence intervals computed on the
held-out test set with 1,000 resamples and seed 42.

## In-hospital mortality (PhysioNet/CinC Challenge 2012, Set A, 48h window)

Primary benchmarks on PhysioNet 2012 Set A — approximately 4,000 ICU
patients with publicly released outcome labels. Sets B and C had their
outcomes withheld by PhysioNet for evaluation purposes and are not used.

| Model                | AUROC | AUPRC | Brier ↓ |
| -------------------- | ----- | ----- | ------- |
| Logistic regression  | TBD   | TBD   | TBD     |
| XGBoost              | TBD   | TBD   | TBD     |
| LSTM                 | TBD   | TBD   | TBD     |
| MedGemma 4B + LoRA   | TBD   | TBD   | TBD     |

Numbers are updated as each model run completes.

## Sanity check: pipeline on synthetic data

A synthetic data generator is included so the pipeline runs end-to-end
without any data downloads. The synthetic data is designed to be largely
separable — its purpose is to verify the implementation, not to estimate
real-world performance. Synthetic-data metrics are not reported here as
research results.

## Reproducibility

Every result reported here can be reproduced by:

```bash
# 1. Download the data (~6MB)
python -m clinical_llm.data.physionet2012_downloader

# 2. Convert to the project's standard schema
python -m clinical_llm.data.physionet2012_loader

# 3. Train any model
python -m clinical_llm.training.train \
    --model <logreg|xgboost|lstm|llm> \
    --data-dir data/physionet2012 \
    --out-dir outputs/<model_name> \
    --seed 42
```

with the corresponding YAML config in `configs/`.

## Calibration

For each model, we report:

- **Brier score** as a summary calibration measure.
- **Calibration curve** (binned reliability diagram) — added with full
  results after training runs complete.
- **Expected calibration error** (ECE) on a 10-bin discretisation —
  added with full results after training runs complete.

## External validation (MIMIC-IV)

MIMIC-IV is supported as an optional external validation cohort for
users with [credentialed access](https://physionet.org/about/credentialing/).
The same training pipeline accepts MIMIC-IV data converted to the
project's standard schema. External validation results will be added
to this document as they become available.
