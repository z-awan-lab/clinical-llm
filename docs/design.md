# Design decisions

This document captures the *why* behind the architectural choices in this
project. It's written as much for future-me as for readers.

## Why MIMIC-IV?

MIMIC-IV is the de facto standard for clinical ML benchmarks. Choosing it
makes results directly comparable to a large existing literature, and it
signals to readers that the project takes evaluation seriously.

The credentialing requirement is a feature, not a bug: it forces serious
treatment of clinical data governance, which is a transferable skill.

## Why in-hospital mortality?

In-hospital mortality from the first 48 hours of ICU data is the canonical
MIMIC benchmark task. Its advantages:

- **Comparable**: virtually every clinical ML paper benchmarks against it.
- **Well-defined**: the label is binary, observable, and unambiguous.
- **Clinically meaningful**: high-utility, low-leakage prediction.
- **Reasonable class imbalance**: around 10% positive — challenging but
  not pathological.

Other tasks (length of stay, readmission, phenotyping) are on the roadmap
once mortality is fully landed.

## Why synthetic data as a first-class citizen?

Three reasons:

1. **Recruiters and reviewers must be able to run the code.** Most won't
   have MIMIC credentials, but they will judge the pipeline. A repo that
   cannot be executed without a multi-week approval process is a repo that
   doesn't get evaluated.
2. **CI requires it.** GitHub Actions runners cannot legally hold MIMIC.
3. **It surfaces schema bugs early.** A schema-mirroring generator catches
   "my code assumed the column was a string" before real-data integration.

The synthetic data is *not* intended to be clinically realistic; it is
intended to be structurally identical to MIMIC-IV.

## Why patient-level splits?

A patient's events appearing in both train and test would leak information
through correlated measurements within the same trajectory. This inflates
metrics in a way that has tripped up published work. The split utility
enforces patient-level partitioning at the API level — the alternative is
hard to do wrong here.

## Why bootstrap CIs on every metric?

Clinical datasets are often small. A point estimate without uncertainty
quantification is a misleading point estimate. Bootstrap CIs are cheap to
compute and the standard expectation in clinical ML reporting.

## Why an honest baseline ladder?

The story of the project is "an LLM beats simple baselines on clinical
sequences" — *if and only if* that is true. The ladder (LogReg → XGBoost →
LSTM → LLM) is designed so that any failure to improve is visible and
honest. If the LLM only marginally beats XGBoost, the project reports
that and discusses why.

The temptation in portfolio projects is to pick comparison points the
flagship model is guaranteed to win against. This project resists that.

## Why Llama-3.2-3B specifically?

- Small enough to fine-tune on a single GPU with LoRA.
- Open weights, permissive licence.
- Recent enough to be of interest, not so new as to be unstable.
- 3B is the sweet spot for "real LLM" without becoming "lab-only LLM".

Alternatives considered: Phi-3-mini (also reasonable; will be a comparison
later), Mistral-7B (heavier, harder to fine-tune on commodity GPUs),
BioMedLM (domain-pretrained but smaller community).

## Why LoRA?

Full fine-tuning of a 3B model on a small clinical cohort would overfit and
require many more GPU hours. LoRA fine-tunes a small adapter, keeps the
base model frozen, and is the modern standard for this size of task. It is
also what every industry team uses, so it's a recruiter-relevant skill.

## Why MedGemma 4B?

We default to `google/medgemma-4b-it` rather than a general-purpose base
model like Llama-3.2-3B because MedGemma is pretrained on a large
medical corpus. For a clinical prediction task, this typically gives:

  * Better sample efficiency — the tokeniser and pretraining have
    already seen the long tail of clinical vocabulary.
  * A more interesting comparison point — "did a medical LLM beat a
    general LLM beat XGBoost?" is a richer story than just "did an LLM
    beat XGBoost?"

The model identifier is configurable via `--model-name`, so the same
training pipeline can produce results for Llama-3.2-3B, Phi-3-mini, or
any HF causal LM. The licence (Gemma terms of use) permits research and
non-clinical-deployment use; this code is research/benchmarking only.

## Why LoRA?

Full fine-tuning of a 4B model on a small clinical cohort would overfit
and require many more GPU hours. LoRA fine-tunes a small adapter, keeps
the base model frozen, and is the modern standard for this size of task.
It is also what every industry team uses, so it's a recruiter-relevant
skill.

## Why no LLM smoke test in CI?

The CI runners on GitHub Actions are CPU-only and don't have the memory
or time budget to load MedGemma 4B. We instead run the LLM module's
structural tests (config validation, interface checks) in CI, and run
the actual training on a GPU off-CI. This is the standard pattern for
ML CI — fast structural checks always, heavy integration runs on
dedicated infrastructure.

## Why no W&B / MLflow / Hydra?

Configuration is intentionally light (YAML files in `configs/`) to keep the
project understandable in a single sitting. A production team would
benefit from heavier tooling; a portfolio project is harmed by it because
it raises the bar to "I can read this codebase in 20 minutes" — which is
exactly the bar a recruiter scanning the repo applies.

These can be added later if the project graduates beyond portfolio scope.
