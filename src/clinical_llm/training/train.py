"""End-to-end training script for tabular baselines.

This is the unified entry point for all baselines that consume the
flat feature matrix (logistic regression and XGBoost). It:

    1. Loads (or generates) data
    2. Extracts features
    3. Splits patient-wise
    4. Fits the chosen model
    5. Evaluates on val and test with bootstrap CIs
    6. Saves artifacts (results JSON + per-model importances)

Run:
    python -m clinical_llm.training.train --model logreg
    python -m clinical_llm.training.train --model xgboost
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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

MODELS = {"logreg", "xgboost"}


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patients and events tables from CSVs."""
    patients = pd.read_csv(data_dir / "patients.csv")
    events = pd.read_csv(data_dir / "events.csv", parse_dates=["charttime"])
    return patients, events


def _default_out_dir(model: str) -> Path:
    return Path(f"outputs/baseline_{model}")


def _train_logreg(X_train, y_train, X_val, y_val, seed: int):
    model = LogisticRegressionBaseline(LogRegConfig(seed=seed))
    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train, y_train, X_val, y_val, seed: int):
    model = XGBoostBaseline(XGBoostConfig(seed=seed))
    # XGBoost uses the validation set for early stopping.
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    return model


TRAINERS = {
    "logreg": _train_logreg,
    "xgboost": _train_xgboost,
}


def _save_importances(model, out_dir: Path, model_name: str) -> None:
    """Save the model's feature importances or coefficients."""
    if model_name == "logreg":
        model.coefficients.to_csv(out_dir / "coefficients.csv", header=["coefficient"])
    elif model_name == "xgboost":
        model.feature_importances.to_csv(out_dir / "feature_importances.csv", header=["gain"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS), default="logreg")
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

    print("Extracting features...")
    train_events = filter_events_by_patients(events, train_pts)
    val_events = filter_events_by_patients(events, val_pts)
    test_events = filter_events_by_patients(events, test_pts)

    X_train = extract_features(train_pts, train_events)
    X_val = extract_features(val_pts, val_events)
    X_test = extract_features(test_pts, test_events)

    label_col = get_label_column()
    y_train = train_pts.set_index("patient_id").loc[X_train["patient_id"], label_col].values
    y_val = val_pts.set_index("patient_id").loc[X_val["patient_id"], label_col].values
    y_test = test_pts.set_index("patient_id").loc[X_test["patient_id"], label_col].values

    feature_cols = [c for c in feature_columns(X_train) if c != "patient_id"]
    X_train_feat = X_train[feature_cols]
    X_val_feat = X_val[feature_cols]
    X_test_feat = X_test[feature_cols]

    print(f"Training {args.model} on {len(feature_cols)} features...")
    trainer = TRAINERS[args.model]
    model = trainer(X_train_feat, y_train, X_val_feat, y_val, args.seed)

    print("\nValidation set:")
    val_results = evaluate(y_val, model.predict_proba(X_val_feat), seed=args.seed)
    print(val_results.summary())

    print("\nTest set:")
    test_results = evaluate(y_test, model.predict_proba(X_test_feat), seed=args.seed)
    print(test_results.summary())

    # Persist artifacts.
    results_payload = {
        "model": args.model,
        "seed": args.seed,
        "n_features": len(feature_cols),
        "validation": val_results.to_dict(),
        "test": test_results.to_dict(),
    }
    (out_dir / "results.json").write_text(json.dumps(results_payload, indent=2))
    _save_importances(model, out_dir, args.model)

    print(f"\nArtifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
