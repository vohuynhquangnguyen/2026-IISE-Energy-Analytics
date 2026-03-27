"""
main.py
=======
Entry point for the IISE Energy Analytics competition pipeline.

The pipeline is divided into three clearly separated phases:

    Phase 1 — Configuration & Data Loading
        Load YAML configs, read the NetCDF competition datasets, and print
        a summary of the data.

    Phase 2 — Walk-Forward Cross-Validation
        For each enabled model, run *K* walk-forward folds (expanding-window
        temporal CV) to evaluate forecasting performance and tune
        hyperparameters without look-ahead bias.

    Phase 3 — Test Inference
        Retrain each model on the *full* training set and produce the 48-hour
        forecast file required for competition submission.

All feature engineering is centralised in ``utils/feature_engineering.py`` so
that every model trains and validates on an identical feature set.

Reference: demo.ipynb (organiser-provided baseline notebook).
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
from models.sarimax import CountySARIMAX
from models.seq2seq import Seq2SeqForecaster
from models.dkl import DKLForecaster


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


def add_quantile_intervals(
    pred_df: pd.DataFrame,
    q_by_county: np.ndarray,
    locations: list[str],
) -> pd.DataFrame:
    """
    Attach lower/upper columns to a long prediction DataFrame using
    county-specific absolute-residual quantiles (conformal calibration).
    """
    q_map = {str(loc): float(q_by_county[i]) for i, loc in enumerate(locations)}
    out = pred_df.copy()
    out["location"] = out["location"].astype(str)
    q_vals = out["location"].map(q_map).astype(float)
    out["lower"] = np.clip(out["pred"] - q_vals, 0, None)
    out["upper"] = out["pred"] + q_vals
    return out


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

def validate_sarimax_fold(
    fold: WalkForwardFold,
    locations: list[str],
    model_cfg_path: str = "sarimax.yaml",
) -> tuple[pd.DataFrame, CompetitionScores]:
    """Train SARIMAX on one fold and return predictions + scores."""
    model = CountySARIMAX.from_config(model_cfg_path)
    model.fit(fold.ds_train_sub)

    # Pass ds_future=fold.ds_val so that exogenous weather features
    # can be extracted for the validation forecast horizon.
    pred_df = model.predict(
        fold.val_timestamps, locations,
        return_intervals=True, alpha=model.alpha,
        ds_future=fold.ds_val,
    )

    demo_rmse_val, scores = compute_competition_scores(fold, pred_df, locations)
    print_scores("SARIMAX", demo_rmse_val, scores)
    return pred_df, scores


def validate_seq2seq_fold(
    fold: WalkForwardFold,
    locations: list[str],
    model_cfg_path: str = "seq2seq.yaml",
) -> tuple[pd.DataFrame, CompetitionScores]:
    """Train Seq2Seq on one fold and return predictions + scores."""
    cfg = load_config(model_cfg_path)
    interval_cfg = cfg.get("intervals", {})
    cal_size = interval_cfg.get("cal_size", 48)
    q_level = interval_cfg.get("q_level", 0.95)

    model = Seq2SeqForecaster.from_config(model_cfg_path)

    # --- Conformal calibration: hold out trailing block from fold's train set ---
    ds_train = fold.ds_train_sub
    n_train = ds_train.sizes["timestamp"]

    if n_train > cal_size + model.seq_len + model.horizon:
        # Split fold's train into fit + calibration
        ds_fit = ds_train.isel(timestamp=slice(0, n_train - cal_size))
        ds_cal = ds_train.isel(timestamp=slice(n_train - cal_size, n_train))

        cal_timestamps = pd.to_datetime(ds_cal.timestamp.values[: model.horizon])
        cal_truth = (
            ds_cal.out.transpose("timestamp", "location")
            .isel(timestamp=slice(0, model.horizon))
            .values.astype(float)
        )

        # Train calibration model and measure residuals
        cal_model = Seq2SeqForecaster.from_config(model_cfg_path)
        cal_model.fit(ds_fit)
        cal_pred_df = cal_model.predict(ds_fit, cal_timestamps)
        cal_pred = long_df_to_matrix(
            cal_pred_df, value_col="pred",
            timestamps=cal_timestamps, locations=locations,
        )
        abs_resid = np.abs(cal_truth - cal_pred)
        q_by_county = np.quantile(abs_resid, q_level, axis=0)
    else:
        # Not enough data for calibration — fall back to no intervals
        q_by_county = None

    # --- Train the actual model on the full fold training set ---
    model.fit(fold.ds_train_sub)
    pred_df = model.predict(fold.ds_train_sub, fold.val_timestamps)

    if q_by_county is not None:
        pred_df = add_quantile_intervals(pred_df, q_by_county, locations)

    demo_rmse_val, scores = compute_competition_scores(fold, pred_df, locations)
    print_scores("Seq2Seq", demo_rmse_val, scores)
    return pred_df, scores


def validate_dkl_fold(
    fold: WalkForwardFold,
    locations: list[str],
    model_cfg_path: str = "dkl.yaml",
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, CompetitionScores]:
    """Train DKL on one fold and return predictions + scores."""
    model = DKLForecaster.from_config(model_cfg_path)
    model.fit(fold.ds_train_sub)

    # Point predictions (autoregressive)
    pred_df = model.predict(
        fold.ds_train_sub, fold.val_timestamps, return_intervals=False,
    )

    # Apply adaptive conformal intervals on the point predictions
    _, pred_matrix = average_county_rmse(
        fold.val_truth, pred_df, locations, fold.val_timestamps,
    )

    y_true_flat = fold.val_truth.ravel()
    pred_flat = pred_matrix.ravel()

    residuals = np.abs(y_true_flat - pred_flat)
    n_cal = len(residuals)
    conformal_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    conformal_q = np.quantile(residuals, conformal_level)

    # Fixed conformal
    conf_lo = np.clip(pred_flat - conformal_q, 0, None).reshape(pred_matrix.shape)
    conf_hi = (pred_flat + conformal_q).reshape(pred_matrix.shape)

    # Adaptive conformal (wider intervals for high-outage predictions)
    median_nz = np.median(pred_flat[pred_flat > 0]) if (pred_flat > 0).any() else 1.0
    adapt_q = np.where(pred_flat > median_nz, conformal_q * 1.5, conformal_q * 0.8)
    adapt_q = np.maximum(adapt_q, 5.0)
    adapt_lo = np.clip(pred_flat - adapt_q, 0, None).reshape(pred_matrix.shape)
    adapt_hi = (pred_flat + adapt_q).reshape(pred_matrix.shape)

    # Pick the strategy with best Winkler among those meeting coverage ≥ 94%
    def _wk(lo, hi, yt):
        return np.mean(
            (hi - lo)
            + (2 / alpha) * np.maximum(lo - yt, 0)
            + (2 / alpha) * np.maximum(yt - hi, 0)
        )

    conf_cov = np.mean((y_true_flat >= conf_lo.ravel()) & (y_true_flat <= conf_hi.ravel()))
    adapt_cov = np.mean((y_true_flat >= adapt_lo.ravel()) & (y_true_flat <= adapt_hi.ravel()))
    conf_wk = _wk(conf_lo.ravel(), conf_hi.ravel(), y_true_flat)
    adapt_wk = _wk(adapt_lo.ravel(), adapt_hi.ravel(), y_true_flat)

    print(f"  PI strategies: Fixed(cov={conf_cov:.3f}, wk={conf_wk:.1f}) "
          f"| Adaptive(cov={adapt_cov:.3f}, wk={adapt_wk:.1f})")

    if adapt_cov >= 0.94 and adapt_wk < conf_wk:
        lower_matrix, upper_matrix = adapt_lo, adapt_hi
        print("  → Using: Adaptive Conformal")
    else:
        lower_matrix, upper_matrix = conf_lo, conf_hi
        print("  → Using: Conformal (fixed)")

    # Rebuild long-format DataFrame with intervals
    train_outages = (
        fold.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )
    scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=fold.val_truth,
        y_pred=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
    )
    demo_rmse_val = float(np.nanmean([
        rmse(fold.val_truth[:, i], pred_matrix[:, i])
        for i in range(len(locations))
    ]))
    print_scores("DKL", demo_rmse_val, scores)

    return pred_df, scores


# =========================================================================== #
#  Walk-forward validation driver                                              #
# =========================================================================== #

def run_walk_forward_validation(
    folds: list[WalkForwardFold],
    locations: list[str],
    enabled_models: list[str],
    alpha: float = 0.05,
) -> dict[str, list[CompetitionScores]]:
    """
    Run walk-forward cross-validation for all enabled models.

    Parameters
    ----------
    folds : list[WalkForwardFold]
        Temporal folds generated by ``walk_forward_split``.
    locations : list[str]
        County identifiers.
    enabled_models : list[str]
        Which models to run (e.g. ["sarimax", "seq2seq", "dkl"]).
    alpha : float
        Significance level for prediction intervals.

    Returns
    -------
    dict[str, list[CompetitionScores]]
        Model name → list of per-fold scores.
    """
    all_scores: dict[str, list[CompetitionScores]] = {m: [] for m in enabled_models}

    for fold in folds:
        print_fold_header(fold)

        if "sarimax" in enabled_models:
            print("\n  Training SARIMAX...")
            _, scores = validate_sarimax_fold(fold, locations)
            all_scores["sarimax"].append(scores)

        if "seq2seq" in enabled_models:
            print("\n  Training Seq2Seq...")
            _, scores = validate_seq2seq_fold(fold, locations)
            all_scores["seq2seq"].append(scores)

        if "dkl" in enabled_models:
            print("\n  Training DKL...")
            _, scores = validate_dkl_fold(fold, locations, alpha=alpha)
            all_scores["dkl"].append(scores)

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

def retrain_and_predict_sarimax(
    bundle: CompetitionData,
    results_dir: Path,
) -> None:
    """Retrain SARIMAX on the full training set and save test predictions."""
    if bundle.ds_test_48h is None:
        print("[SARIMAX] No test set found — skipping.")
        return

    print("[SARIMAX] Retraining on full training data...")
    model = CountySARIMAX.from_config("sarimax.yaml")
    model.fit(bundle.ds_train)

    # Pass ds_future=bundle.ds_test_48h so that exogenous weather features
    # can be extracted for the 48-hour test forecast horizon.
    pred_df = model.predict(
        bundle.test_48h_timestamps, bundle.locations,
        return_intervals=True, alpha=model.alpha,
        ds_future=bundle.ds_test_48h,
    )
    out_path = results_dir / "sarimax_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[SARIMAX] Saved → {out_path}")


def retrain_and_predict_seq2seq(
    bundle: CompetitionData,
    results_dir: Path,
) -> None:
    """Retrain Seq2Seq on the full training set and save test predictions."""
    if bundle.ds_test_48h is None:
        print("[Seq2Seq] No test set found — skipping.")
        return

    cfg = load_config("seq2seq.yaml")
    interval_cfg = cfg.get("intervals", {})
    cal_size = interval_cfg.get("cal_size", 48)
    q_level = interval_cfg.get("q_level", 0.95)

    # Estimate interval widths via conformal calibration on full train data
    print("[Seq2Seq] Calibrating intervals...")
    model_cfg = Seq2SeqForecaster.from_config("seq2seq.yaml")
    ds_train = bundle.ds_train
    n = ds_train.sizes["timestamp"]

    ds_fit = ds_train.isel(timestamp=slice(0, n - cal_size))
    ds_cal = ds_train.isel(timestamp=slice(n - cal_size, n))

    cal_timestamps = pd.to_datetime(ds_cal.timestamp.values[: model_cfg.horizon])
    cal_truth = (
        ds_cal.out.transpose("timestamp", "location")
        .isel(timestamp=slice(0, model_cfg.horizon))
        .values.astype(float)
    )

    cal_model = Seq2SeqForecaster.from_config("seq2seq.yaml")
    cal_model.fit(ds_fit)
    cal_pred_df = cal_model.predict(ds_fit, cal_timestamps)
    cal_pred = long_df_to_matrix(
        cal_pred_df, value_col="pred",
        timestamps=cal_timestamps, locations=bundle.locations,
    )
    q_by_county = np.quantile(np.abs(cal_truth - cal_pred), q_level, axis=0)

    # Train on full data and predict
    print("[Seq2Seq] Retraining on full training data...")
    model = Seq2SeqForecaster.from_config("seq2seq.yaml")
    model.fit(bundle.ds_train)

    pred_df = model.predict(bundle.ds_train, bundle.test_48h_timestamps)
    pred_df = add_quantile_intervals(pred_df, q_by_county, bundle.locations)

    out_path = results_dir / "seq2seq_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[Seq2Seq] Saved → {out_path}")


def retrain_and_predict_dkl(
    bundle: CompetitionData,
    results_dir: Path,
) -> None:
    """Retrain DKL on the full training set and save test predictions."""
    if bundle.ds_test_48h is None:
        print("[DKL] No test set found — skipping.")
        return

    dkl_cfg = load_config("dkl.yaml")
    interval_cfg = dkl_cfg.get("intervals", {})

    print("[DKL] Retraining on full training data...")
    model = DKLForecaster.from_config("dkl.yaml")
    model.fit(bundle.ds_train)

    if interval_cfg.get("method", "conformal") == "conformal":
        print("[DKL] Calibrating conformal intervals...")
        model.calibrate_intervals(
            bundle.ds_train,
            calibration_size=interval_cfg.get("calibration_size", 48),
            alpha=interval_cfg.get("alpha", 0.05),
        )

    pred_df = model.predict(
        bundle.ds_train, bundle.test_48h_timestamps,
        return_intervals=True,
        alpha=interval_cfg.get("alpha", 0.05),
        interval_method=interval_cfg.get("method", "conformal"),
    )

    out_path = results_dir / "dkl_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[DKL] Saved → {out_path}")


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
    enabled_models = pipeline_cfg.get("models", ["sarimax", "seq2seq", "dkl"])
    alpha = pipeline_cfg.get("intervals", {}).get("alpha", 0.05)

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
    print(f"    Enabled models:   {enabled_models}")
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
        enabled_models=enabled_models,
        alpha=alpha,
    )

    print_cv_summary(all_scores)

    # ================================================================== #
    #  Phase 3: Test Inference                                             #
    # ================================================================== #
    print("=" * 70)
    print("  Phase 3: Test Inference (retrain on full training data)")
    print("=" * 70)
    print()

    if "sarimax" in enabled_models:
        retrain_and_predict_sarimax(bundle, results_dir)

    if "seq2seq" in enabled_models:
        retrain_and_predict_seq2seq(bundle, results_dir)

    if "dkl" in enabled_models:
        retrain_and_predict_dkl(bundle, results_dir)

    print("\n" + "=" * 70)
    print("  Pipeline complete.  Results saved to:", results_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
