# Results

> Results will be updated as each model on the roadmap completes training.
> All metrics report bootstrap 95% confidence intervals computed on the
> held-out test set with 1,000 resamples and seed 42.

## In-hospital mortality (MIMIC-IV, 48h observation window)

| Model                | AUROC | AUPRC | Brier ↓ |
| -------------------- | ----- | ----- | ------- |
| Logistic regression  | TBD   | TBD   | TBD     |
| XGBoost              | TBD   | TBD   | TBD     |
| LSTM                 | TBD   | TBD   | TBD     |
| Llama-3.2-3B + LoRA  | TBD   | TBD   | TBD     |

## Sanity check: pipeline on synthetic data

The pipeline first runs on synthetic data designed to be linearly
separable, primarily to verify the implementation. Performance on
synthetic data is **not** indicative of real-data performance and is not
reported here.

## Reproducibility

Every result reported here can be reproduced by:

```bash
python -m clinical_llm.training.train \
    --data-dir data/mimic_iv \
    --out-dir outputs/<model_name> \
    --seed 42
```

with the corresponding YAML config in `configs/`.

## Calibration

Calibration plots will be added once real-data runs complete. We report:

- **Calibration curve** (binned reliability diagram) with 95% CIs.
- **Brier score** as a summary measure.
- **Expected calibration error** (ECE) on a 10-bin discretisation.
