"""
main.py
=======
Entry point for the IISE Energy Analytics competition pipeline.

The pipeline is divided into three clearly separated phases:

    Phase 1 — Configuration & Data Loading
        Load YAML configs, read the NetCDF competition datasets, and print
        a summary of the data.

    Phase 2 — Walk-Forward Cross-Validation
        Run *K* walk-forward folds (expanding-window temporal CV) to evaluate 
        forecasting performance and tune hyperparameters without look-ahead bias 
        for the XGBoostLSS model.

    Phase 3 — Test Inference
        Retrain the XGBoostLSS model on the *full* training set and produce the 48-hour
        forecast file required for competition submission.

All feature engineering is centralised in ``utils/feature_engineering.py`` so
that every model trains and validates on an identical feature set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from utils.data_loader import (
    CompetitionData,
    WalkForwardFold,
    load_competition_data,
    walk_forward_split,
)
from evaluation.metrics import (
    evaluate_all_metrics,
    long_df_to_matrix,
    rmse,
    CompetitionScores,
)

# Only import the XGBoostLSS model
from models.xgboostlss import XGBoostLSSForecaster


# =========================================================================== #
#  Utility helpers                                                             #
# =========================================================================== #

def average_county_rmse(
    y_true_matrix: np.ndarray,
    pred_df: pd.DataFrame,
    locations: list[str],
    timestamps: pd.DatetimeIndex,
) -> tuple[float, np.ndarray]:
    """
    Compute the macro-averaged county-level RMSE (demo-notebook style).

    Returns the average RMSE and the prediction matrix (T × C).
    """
    pred_matrix = long_df_to_matrix(
        pred_df, value_col="pred", timestamps=timestamps, locations=locations,
    )
    county_rmses = [
        rmse(y_true_matrix[:, i], pred_matrix[:, i])
        for i in range(len(locations))
    ]
    return float(np.nanmean(county_rmses)), pred_matrix


def print_fold_header(fold: WalkForwardFold) -> None:
    """Print a concise header for a walk-forward fold."""
    n_train = fold.ds_train_sub.sizes["timestamp"]
    n_val = len(fold.val_timestamps)
    print(f"\n{'─' * 60}")
    print(f"  Fold {fold.fold_idx}  │  train={n_train} ts  │  val={n_val} ts")
    print(f"  val range: {fold.val_timestamps[0]} → {fold.val_timestamps[-1]}")
    print(f"{'─' * 60}")


def print_scores(model_name: str, demo_rmse: float, scores: CompetitionScores) -> None:
    """Pretty-print the competition metrics for one model on one fold."""
    print(f"  [{model_name}] Avg county RMSE: {demo_rmse:.4f}")
    print(f"    s1 Normal RMSE:  {scores.s1_normal_rmse:.4f}")
    print(f"    s2 Tail RMSE:    {scores.s2_tail_rmse:.4f}")
    print(f"    s3 F1 (nonzero): {scores.s3_f1_nonzero:.4f}")
    if np.isnan(scores.s4_winkler_95):
        print("    s4 Winkler 95:   NaN (no intervals)")
    else:
        print(f"    s4 Winkler 95:   {scores.s4_winkler_95:.4f}")


def compute_competition_scores(
    fold: WalkForwardFold,
    pred_df: pd.DataFrame,
    locations: list[str],
) -> tuple[float, CompetitionScores]:
    """
    Evaluate a model's predictions on one walk-forward fold.

    Returns (avg_county_rmse, CompetitionScores).
    """
    demo_rmse_val, pred_matrix = average_county_rmse(
        fold.val_truth, pred_df, locations, fold.val_timestamps,
    )

    # Training outages needed for threshold computation (s1/s2)
    train_outages = (
        fold.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    # Extract interval matrices if present
    lower_matrix = upper_matrix = None
    if "lower" in pred_df.columns and "upper" in pred_df.columns:
        lower_matrix = long_df_to_matrix(
            pred_df, "lower", fold.val_timestamps, locations,
        )
        upper_matrix = long_df_to_matrix(
            pred_df, "upper", fold.val_timestamps, locations,
        )

    scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=fold.val_truth,
        y_pred=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
    )
    return demo_rmse_val, scores


# =========================================================================== #
#  Per-model validation on a single fold                                       #
# =========================================================================== #

def validate_xgboostlss_fold(
    fold: WalkForwardFold,
    locations: list[str],
    model_cfg_path: str = "xgboostlss.yaml",
) -> tuple[pd.DataFrame, CompetitionScores]:
    """Train XGBoostLSS on one fold and return predictions + scores."""
    model = XGBoostLSSForecaster.from_config(model_cfg_path)
    model.fit(fold.ds_train_sub)

    pred_df = model.predict(
        fold.ds_train_sub, fold.val_timestamps, return_intervals=True,
    )

    demo_rmse_val, scores = compute_competition_scores(fold, pred_df, locations)
    print_scores("XGBoostLSS", demo_rmse_val, scores)
    return pred_df, scores


# =========================================================================== #
#  Walk-forward validation driver                                              #
# =========================================================================== #

def run_walk_forward_validation(
    folds: list[WalkForwardFold],
    locations: list[str],
) -> dict[str, list[CompetitionScores]]:
    """
    Run walk-forward cross-validation for the XGBoostLSS model.
    """
    all_scores: dict[str, list[CompetitionScores]] = {"xgboostlss": []}

    for fold in folds:
        print_fold_header(fold)
        print("\n  Training XGBoostLSS...")
        _, scores = validate_xgboostlss_fold(fold, locations)
        all_scores["xgboostlss"].append(scores)

    return all_scores


def print_cv_summary(all_scores: dict[str, list[CompetitionScores]]) -> None:
    """Print a summary table of walk-forward CV results averaged across folds."""
    print("\n" + "=" * 70)
    print("  WALK-FORWARD CROSS-VALIDATION SUMMARY (mean ± std across folds)")
    print("=" * 70)

    for model_name, fold_scores in all_scores.items():
        if not fold_scores:
            continue
        s1 = [s.s1_normal_rmse for s in fold_scores]
        s2 = [s.s2_tail_rmse for s in fold_scores]
        s3 = [s.s3_f1_nonzero for s in fold_scores]
        s4 = [s.s4_winkler_95 for s in fold_scores if not np.isnan(s.s4_winkler_95)]

        print(f"\n  {model_name.upper()} ({len(fold_scores)} folds)")
        print(f"    s1 Normal RMSE:  {np.mean(s1):.4f} ± {np.std(s1):.4f}")
        print(f"    s2 Tail RMSE:    {np.mean(s2):.4f} ± {np.std(s2):.4f}")
        print(f"    s3 F1 (nonzero): {np.mean(s3):.4f} ± {np.std(s3):.4f}")
        if s4:
            print(f"    s4 Winkler 95:   {np.mean(s4):.4f} ± {np.std(s4):.4f}")
        else:
            print("    s4 Winkler 95:   N/A")
    print()


# =========================================================================== #
#  Phase 3: Test inference helpers                                             #
# =========================================================================== #

def retrain_and_predict_xgboostlss(
    bundle: CompetitionData,
    results_dir: Path,
) -> None:
    """Retrain XGBoostLSS on the full training set and save test predictions."""
    if bundle.ds_test_48h is None:
        print("[XGBoostLSS] No test set found — skipping.")
        return

    print("[XGBoostLSS] Retraining on full training data...")
    model = XGBoostLSSForecaster.from_config("xgboostlss.yaml")
    model.fit(bundle.ds_train)

    pred_df = model.predict(
        bundle.ds_train, bundle.test_48h_timestamps, return_intervals=True,
    )

    out_path = results_dir / "xgboostlss_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[XGBoostLSS] Saved → {out_path}")


# =========================================================================== #
#  Main entry point                                                            #
# =========================================================================== #

def main() -> None:
    """
    Full pipeline: config → data → walk-forward CV → test inference.
    """
    # ================================================================== #
    #  Phase 1: Configuration & Data Loading                              #
    # ================================================================== #
    print("=" * 70)
    print("  Phase 1: Configuration & Data Loading")
    print("=" * 70)

    pipeline_cfg = load_config("pipeline.yaml")
    data_cfg = pipeline_cfg.get("data", {})
    wf_cfg = pipeline_cfg.get("walk_forward", {})

    results_dir = Path(pipeline_cfg.get("results_dir", "results"))
    results_dir.mkdir(exist_ok=True)

    # Load competition data
    bundle = load_competition_data(
        data_dir=data_cfg.get("data_dir", "data"),
        train_file=data_cfg.get("train_file", "train.nc"),
        test_48h_file=data_cfg.get("test_48h_file", "test_48h_demo.nc"),
    )

    print(f"\n  Dataset summary")
    print(f"    Train timestamps: {bundle.ds_train.sizes['timestamp']}")
    print(f"    Counties:         {bundle.ds_train.sizes['location']}")
    print(f"    Weather features: {bundle.ds_train.sizes['feature']}")
    print(f"    Enabled model:    XGBoostLSS")
    print()

    # ================================================================== #
    #  Phase 2: Walk-Forward Cross-Validation                             #
    # ================================================================== #
    print("=" * 70)
    print("  Phase 2: Walk-Forward Cross-Validation")
    print("=" * 70)

    folds = walk_forward_split(
        bundle.ds_train,
        n_folds=wf_cfg.get("n_folds", 3),
        horizon=wf_cfg.get("horizon", 48),
        min_train_frac=wf_cfg.get("min_train_frac", 0.5),
    )
    print(f"\n  Created {len(folds)} walk-forward folds "
          f"(horizon={wf_cfg.get('horizon', 48)} hours)")

    all_scores = run_walk_forward_validation(
        folds=folds,
        locations=bundle.locations,
    )

    print_cv_summary(all_scores)

    # ================================================================== #
    #  Phase 3: Test Inference                                             #
    # ================================================================== #
    print("=" * 70)
    print("  Phase 3: Test Inference (retrain on full training data)")
    print("=" * 70)
    print()

    retrain_and_predict_xgboostlss(bundle, results_dir)

    print("\n" + "=" * 70)
    print("  Pipeline complete.  Results saved to:", results_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()