"""End-to-end training script for all baselines.

Unified entry point that branches internally between two paths:

  * Tabular path: logistic regression and XGBoost, operating on a flat
    feature matrix derived from aggregated vital signs.
  * Sequence path: LSTM (and future models), operating on per-patient
    time-aligned tensors of raw vitals with a missingness mask.

The branching is hidden behind a single `--model` flag so the user
experience stays uniform across baselines.

Run:
    python -m clinical_llm.training.train --model logreg
    python -m clinical_llm.training.train --model xgboost
    python -m clinical_llm.training.train --model lstm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_llm.data.features import (
    extract_features,
    feature_columns,
    get_label_column,
)
from clinical_llm.data.splits import (
    describe_split,
    filter_events_by_patients,
    make_splits,
)
from clinical_llm.evaluation.metrics import evaluate
from clinical_llm.models.baselines import (
    LogisticRegressionBaseline,
    LogRegConfig,
    XGBoostBaseline,
    XGBoostConfig,
)

TABULAR_MODELS = {"logreg", "xgboost"}
SEQUENCE_MODELS = {"lstm"}
ALL_MODELS = TABULAR_MODELS | SEQUENCE_MODELS


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patients and events tables from CSVs."""
    patients = pd.read_csv(data_dir / "patients.csv")
    events = pd.read_csv(data_dir / "events.csv", parse_dates=["charttime"])
    return patients, events


def _default_out_dir(model: str) -> Path:
    return Path(f"outputs/baseline_{model}")


# --------------------------------------------------------------------------- #
# Tabular path                                                                #
# --------------------------------------------------------------------------- #


def _build_tabular_inputs(train_pts, val_pts, test_pts, train_events, val_events, test_events):
    X_train = extract_features(train_pts, train_events)
    X_val = extract_features(val_pts, val_events)
    X_test = extract_features(test_pts, test_events)

    label_col = get_label_column()
    y_train = train_pts.set_index("patient_id").loc[X_train["patient_id"], label_col].values
    y_val = val_pts.set_index("patient_id").loc[X_val["patient_id"], label_col].values
    y_test = test_pts.set_index("patient_id").loc[X_test["patient_id"], label_col].values

    feature_cols = [c for c in feature_columns(X_train) if c != "patient_id"]
    return (
        X_train[feature_cols],
        X_val[feature_cols],
        X_test[feature_cols],
        y_train,
        y_val,
        y_test,
        feature_cols,
    )


def _train_logreg(X_train, y_train, X_val, y_val, seed: int):
    model = LogisticRegressionBaseline(LogRegConfig(seed=seed))
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train, y_train, X_val, y_val, seed: int):
    model = XGBoostBaseline(XGBoostConfig(seed=seed))
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model


TABULAR_TRAINERS = {
    "logreg": _train_logreg,
    "xgboost": _train_xgboost,
}


def _run_tabular(args, train_pts, val_pts, test_pts, train_events, val_events, test_events):
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = _build_tabular_inputs(
        train_pts, val_pts, test_pts, train_events, val_events, test_events
    )

    print(f"Training {args.model} on {len(feature_cols)} features...")
    trainer = TABULAR_TRAINERS[args.model]
    model = trainer(X_train, y_train, X_val, y_val, args.seed)

    val_results = evaluate(y_val, model.predict_proba(X_val), seed=args.seed)
    test_results = evaluate(y_test, model.predict_proba(X_test), seed=args.seed)

    return model, val_results, test_results, len(feature_cols)


def _save_tabular_artifacts(model, out_dir: Path, model_name: str) -> None:
    if model_name == "logreg":
        model.coefficients.to_csv(out_dir / "coefficients.csv", header=["coefficient"])
    elif model_name == "xgboost":
        model.feature_importances.to_csv(out_dir / "feature_importances.csv", header=["gain"])


# --------------------------------------------------------------------------- #
# Sequence path                                                               #
# --------------------------------------------------------------------------- #


def _run_lstm(args, train_pts, val_pts, test_pts, train_events, val_events, test_events):
    # Local imports so torch is only required for sequence models.
    from clinical_llm.data.sequences import (
        SequenceConfig,
        SequenceDataset,
        _build_pivot,  # noqa: PLC2701  (internal helper)
        compute_normalisation_stats,
        n_features,
    )
    from clinical_llm.models.lstm import LSTMBaseline, LSTMConfig

    seq_cfg = SequenceConfig(observation_hours=48)

    # Normalisation stats computed on the training pivot only — avoids leakage.
    train_pivot = _build_pivot(train_events, seq_cfg)
    stats = compute_normalisation_stats(train_pivot)

    train_ds = SequenceDataset(train_pts, train_events, stats, config=seq_cfg)
    val_ds = SequenceDataset(val_pts, val_events, stats, config=seq_cfg)
    test_ds = SequenceDataset(test_pts, test_events, stats, config=seq_cfg)

    print(f"Training lstm on {n_features()} vital channels...")
    model = LSTMBaseline(LSTMConfig(seed=args.seed))
    model.fit(train_ds, val_ds)

    val_probs = model.predict_proba(val_ds)
    test_probs = model.predict_proba(test_ds)
    y_val = np.array(val_ds.labels)
    y_test = np.array(test_ds.labels)

    val_results = evaluate(y_val, val_probs, seed=args.seed)
    test_results = evaluate(y_test, test_probs, seed=args.seed)

    return model, val_results, test_results, n_features()


def _save_sequence_artifacts(model, out_dir: Path) -> None:
    # Save normalisation stats and a model state placeholder.
    # We keep the saved state minimal — full checkpointing isn't worth the
    # added complexity at portfolio scale.
    import torch

    if model.model_ is not None:
        torch.save(model.model_.state_dict(), out_dir / "lstm_state_dict.pt")


# --------------------------------------------------------------------------- #
# Entry point                                                                 #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(ALL_MODELS), default="logreg")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="defaults to outputs/baseline_<model>"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir or _default_out_dir(args.model)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    patients, events = load_data(args.data_dir)
    print(f"  {len(patients):,} patients, {len(events):,} events")

    print("Making patient-level splits...")
    train_pts, val_pts, test_pts = make_splits(patients, seed=args.seed)
    describe_split("train", train_pts)
    describe_split("val", val_pts)
    describe_split("test", test_pts)

    train_events = filter_events_by_patients(events, train_pts)
    val_events = filter_events_by_patients(events, val_pts)
    test_events = filter_events_by_patients(events, test_pts)

    if args.model in TABULAR_MODELS:
        model, val_results, test_results, n_feat = _run_tabular(
            args, train_pts, val_pts, test_pts, train_events, val_events, test_events
        )
        _save_tabular_artifacts(model, out_dir, args.model)
    else:
        model, val_results, test_results, n_feat = _run_lstm(
            args, train_pts, val_pts, test_pts, train_events, val_events, test_events
        )
        _save_sequence_artifacts(model, out_dir)

    print("\nValidation set:")
    print(val_results.summary())
    print("\nTest set:")
    print(test_results.summary())

    results_payload = {
        "model": args.model,
        "seed": args.seed,
        "n_features": n_feat,
        "validation": val_results.to_dict(),
        "test": test_results.to_dict(),
    }
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2))

    print(f"\nArtifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
