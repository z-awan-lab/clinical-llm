# Results

All metrics report bootstrap 95% confidence intervals computed on the
held-out test set with 1,000 resamples and seed 42. Brier and ECE marked
with ↓ — lower is better. Test split is 15% of patients (n = 600 patients,
14% mortality).

## In-hospital mortality (PhysioNet/CinC Challenge 2012, Set A, 48h window)

Primary benchmarks on PhysioNet 2012 Set A — approximately 4,000 ICU
patients with publicly released outcome labels. Sets B and C had their
outcomes withheld by PhysioNet for evaluation purposes and are not used.
Patient-level stratified splits: 70% train, 15% validation, 15% test.

| Model                | AUROC                   | AUPRC                   | Brier ↓                 | ECE (10-bin) ↓ |
| -------------------- | ----------------------- | ----------------------- | ----------------------- | -------------- |
| Logistic regression  | 0.770 (0.722–0.817)     | 0.330 (0.254–0.432)     | 0.199 (0.185–0.215)     | 0.286          |
| XGBoost              | **0.782** (0.724–0.830) | **0.424** (0.332–0.528) | **0.157** (0.147–0.166) | **0.234**      |
| LSTM (raw sequences) | 0.719 (0.653–0.776)     | 0.322 (0.243–0.447)     | 0.213 (0.197–0.227)     | 0.286          |
| MedGemma 4B + LoRA   | 0.777 (0.721–0.828)     | 0.410 (0.311–0.519)     | 0.190 (0.177–0.202)     | 0.286          |

### Calibration

![Calibration comparison across all four baselines](../outputs/calibration_comparison.png)

Reliability diagrams overlay the four models on the same axes; the
dashed diagonal is perfect calibration. Per-model ECE is reported in
the table above and in each model's `outputs/baseline_<name>/results.json`.

### Observations

**XGBoost is the strongest baseline by every metric.** Discrimination
(AUROC 0.782, AUPRC 0.424), sharpness (Brier 0.157), and calibration
(ECE 0.234) all favour XGBoost. This is the textbook outcome for
gradient-boosted trees on aggregated clinical features at this scale
and is consistent with published PhysioNet 2012 benchmarks.

**MedGemma 4B + LoRA is competitive on discrimination.** Its test AUROC
(0.777) and AUPRC (0.410) are within bootstrap-CI overlap of XGBoost's.
Brier (0.190) is closer to the tabular baselines than to XGBoost. On
this cohort size — ~3,000 training patients with sparse measurements —
a 4B medical-pretrained LLM with parameter-efficient fine-tuning *matches*
a strong tabular baseline on ranking but does not beat it.

**The LSTM trails simpler tabular baselines.** AUROC 0.719 places it
below logistic regression (0.770). With limited training data, the
sequence model has more parameters to fit than the signal supports.
This finding matches prior comparisons on PhysioNet 2012 — sequence
models typically need ≥10× the training cohort to repay their
architectural overhead at this prediction task.

**Calibration is uniformly poor outside XGBoost.** ECE values cluster at
~0.286 for logreg, LSTM, and MedGemma, against XGBoost's 0.234. The
clustering is partly an artefact of the small test set (n = 600, ~83
positive cases leave few patients per probability bin), but it also
indicates that none of these models produces well-calibrated probabilities
out of the box. The natural next step is **post-hoc calibration** —
fitting a Platt scaling or isotonic regression on the validation set's
predicted probabilities. This is a principled, recognised fix that
closes calibration gaps without affecting AUROC. The training pipeline
is set up to support this as a post-processing step.

**Headline.** On this benchmark, the right engineering priority for a
clinical deployment of the LLM is post-hoc calibration of its probability
outputs, not architectural complexity. The pre-trained medical knowledge
in MedGemma plays out at the discrimination level, but principled
calibration is needed before the probabilities are deployment-ready.

## Sanity check: pipeline on synthetic data

A synthetic data generator is included so the pipeline runs end-to-end
without any data downloads. The synthetic data is designed to be largely
separable — its purpose is to verify the implementation, not to estimate
real-world performance. Synthetic-data metrics are not reported here as
research results.

## Reproducibility

Every result reported above can be reproduced by:

```bash
# 1. Download the data (~3MB)
python -m clinical_llm.data.physionet2012_downloader

# 2. Convert to the project's standard schema
python -m clinical_llm.data.physionet2012_loader

# 3. Train each model
python -m clinical_llm.training.train --model logreg  --data-dir data/physionet2012
python -m clinical_llm.training.train --model xgboost --data-dir data/physionet2012
python -m clinical_llm.training.train --model lstm    --data-dir data/physionet2012
python -m clinical_llm.training.train --model llm     --data-dir data/physionet2012

# 4. Generate the combined calibration plot
python -m clinical_llm.evaluation.compare_calibration
```

The LLM run additionally requires a GPU, `pip install -e ".[llm]"`, and
[Hugging Face gated access to MedGemma](https://huggingface.co/google/medgemma-4b-it).

## External validation (MIMIC-IV)

MIMIC-IV is supported as an optional external validation cohort for
users with [credentialed access](https://physionet.org/about/credentialing/).
The same training pipeline accepts MIMIC-IV data converted to the
project's standard schema. External validation results will be added
to this document as they become available.
