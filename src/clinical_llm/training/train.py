"""End-to-end training script for the logistic regression baseline.

This is the entry point that ties the pipeline together:
    1. Load (or generate) data
    2. Extract features
    3. Split patient-wise
    4. Fit the model
    5. Evaluate on val and test with bootstrap CIs
    6. Save artifacts to disk

Run:
    python -m clinical_llm.training.train --data-dir data/synthetic
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
from clinical_llm.models.baselines import LogisticRegressionBaseline, LogRegConfig


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load patients and events tables from CSVs."""
    patients = pd.read_csv(data_dir / "patients.csv")
    events = pd.read_csv(data_dir / "events.csv", parse_dates=["charttime"])
    return patients, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/baseline_logreg"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    val_events   = filter_events_by_patients(events, val_pts)
    test_events  = filter_events_by_patients(events, test_pts)

    X_train = extract_features(train_pts, train_events)
    X_val   = extract_features(val_pts,   val_events)
    X_test  = extract_features(test_pts,  test_events)

    label_col = get_label_column()
    y_train = train_pts.set_index("patient_id").loc[X_train["patient_id"], label_col].values
    y_val   = val_pts  .set_index("patient_id").loc[X_val  ["patient_id"], label_col].values
    y_test  = test_pts .set_index("patient_id").loc[X_test ["patient_id"], label_col].values

    feature_cols = [c for c in feature_columns(X_train) if c != "patient_id"]
    X_train_feat = X_train[feature_cols]
    X_val_feat   = X_val[feature_cols]
    X_test_feat  = X_test[feature_cols]

    print(f"Training logistic regression on {len(feature_cols)} features...")
    model = LogisticRegressionBaseline(LogRegConfig(seed=args.seed))
    model.fit(X_train_feat, y_train)

    print("\nValidation set:")
    val_results = evaluate(y_val, model.predict_proba(X_val_feat), seed=args.seed)
    print(val_results.summary())

    print("\nTest set:")
    test_results = evaluate(y_test, model.predict_proba(X_test_feat), seed=args.seed)
    print(test_results.summary())

    # Persist artifacts.
    results_payload = {
        "model": "logistic_regression",
        "seed": args.seed,
        "n_features": len(feature_cols),
        "validation": val_results.to_dict(),
        "test": test_results.to_dict(),
    }
    (args.out_dir / "results.json").write_text(json.dumps(results_payload, indent=2))
    model.coefficients.to_csv(args.out_dir / "coefficients.csv", header=["coefficient"])

    print(f"\nArtifacts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
