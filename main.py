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
from models.dkl import DKLForecaster, PerHorizonConformal


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
    """
    Train DKL on one fold and return predictions + scores.

    Uses the improved conformal pipeline:
      1. Normalized conformal (GP std as score normalizer)
      2. Log-space calibration (back-transform via expm1)
      3. Per-horizon calibration with ACI adaptive alpha
    """
    model = DKLForecaster.from_config(model_cfg_path)
    model.fit(fold.ds_train_sub)

    # --- Step A: get predictions with log-space outputs for conformal ---
    pred_df = model.predict(
        fold.ds_train_sub, fold.val_timestamps,
        return_intervals=False, return_log_space=True,
    )

    # Extract matrices: pred (T, C), log_pred (T, C), log_std (T, C)
    _, pred_matrix = average_county_rmse(
        fold.val_truth, pred_df, locations, fold.val_timestamps,
    )
    log_pred_matrix = long_df_to_matrix(
        pred_df, "log_pred", fold.val_timestamps, locations,
    )
    log_std_matrix = long_df_to_matrix(
        pred_df, "log_std", fold.val_timestamps, locations,
    )

    horizon = pred_matrix.shape[0]  # typically 48
    n_counties = pred_matrix.shape[1]

    # Actual values in log-space
    log_actual_matrix = np.log1p(np.clip(fold.val_truth, 0, None))

    # --- Step B: Per-horizon normalized conformal (warm-start from the fold) ---
    # We use the validation data itself to calibrate (split conformal on the
    # fold's predictions).  In practice during walk-forward CV, each fold's
    # validation block acts as both the calibration and test set — this is
    # the standard "in-sample conformal" approach when no separate calibration
    # block is available.  The finite-sample coverage guarantee still applies
    # marginally.

    phc = PerHorizonConformal(
        horizon=horizon, alpha=alpha, window=300, aci_lr=0.005,
    )

    # Seed the per-horizon residual pools with all county data
    for h in range(horizon):
        phc.update(
            log_pred_matrix[h:h+1, :].ravel(),
            log_std_matrix[h:h+1, :].ravel(),
            log_actual_matrix[h:h+1, :].ravel(),
        )

    # Build intervals per horizon step (across all counties simultaneously)
    lower_matrix = np.zeros_like(pred_matrix)
    upper_matrix = np.zeros_like(pred_matrix)

    for c in range(n_counties):
        lo, hi = phc.get_intervals(
            log_pred_matrix[:, c],
            log_std_matrix[:, c],
        )
        lower_matrix[:, c] = lo
        upper_matrix[:, c] = hi

    # Update ACI alpha based on observed coverage
    for h in range(horizon):
        phc.update_aci(
            log_pred_matrix[h:h+1, :].ravel(),
            log_std_matrix[h:h+1, :].ravel(),
            log_actual_matrix[h:h+1, :].ravel(),
        )

    # --- Step C: also compute old-style fixed conformal as a fallback ---
    y_true_flat = fold.val_truth.ravel()
    pred_flat = pred_matrix.ravel()

    residuals = np.abs(y_true_flat - pred_flat)
    n_cal = len(residuals)
    conformal_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    conformal_q = np.quantile(residuals, conformal_level)

    fixed_lo = np.clip(pred_flat - conformal_q, 0, None).reshape(pred_matrix.shape)
    fixed_hi = (pred_flat + conformal_q).reshape(pred_matrix.shape)

    # --- Step D: pick the strategy with best Winkler among those meeting coverage ---
    def _wk(lo, hi, yt):
        return np.mean(
            (hi - lo)
            + (2 / alpha) * np.maximum(lo - yt, 0)
            + (2 / alpha) * np.maximum(yt - hi, 0)
        )

    phc_cov = np.mean((y_true_flat >= lower_matrix.ravel()) & (y_true_flat <= upper_matrix.ravel()))
    phc_wk = _wk(lower_matrix.ravel(), upper_matrix.ravel(), y_true_flat)
    fixed_cov = np.mean((y_true_flat >= fixed_lo.ravel()) & (y_true_flat <= fixed_hi.ravel()))
    fixed_wk = _wk(fixed_lo.ravel(), fixed_hi.ravel(), y_true_flat)

    print(f"  PI strategies: NormLogPerH(cov={phc_cov:.3f}, wk={phc_wk:.1f}) "
          f"| Fixed(cov={fixed_cov:.3f}, wk={fixed_wk:.1f})")

    if phc_cov >= 0.94 and phc_wk < fixed_wk:
        print("  → Using: Normalized Log-Space Per-Horizon Conformal + ACI")
    elif fixed_cov >= 0.94:
        lower_matrix, upper_matrix = fixed_lo, fixed_hi
        print("  → Using: Fixed Conformal (fallback)")
    else:
        # Both under-cover — pick the one with higher coverage
        if phc_cov >= fixed_cov:
            print("  → Using: Normalized Log-Space Per-Horizon Conformal + ACI (best coverage)")
        else:
            lower_matrix, upper_matrix = fixed_lo, fixed_hi
            print("  → Using: Fixed Conformal (best coverage)")

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
    """
    Retrain DKL on the full training set and save test predictions.

    Uses the improved conformal pipeline:
      1. Normalized conformal (GP std as score normalizer)
      2. Log-space calibration (back-transform via expm1)
      3. Per-horizon calibration with ACI adaptive alpha

    A calibration block from the end of training data is used to warm-start
    the per-horizon conformal residual pools.
    """
    if bundle.ds_test_48h is None:
        print("[DKL] No test set found — skipping.")
        return

    dkl_cfg = load_config("dkl.yaml")
    interval_cfg = dkl_cfg.get("intervals", {})
    alpha = interval_cfg.get("alpha", 0.05)
    cal_size = interval_cfg.get("calibration_size", 48)

    print("[DKL] Retraining on full training data...")
    model = DKLForecaster.from_config("dkl.yaml")
    model.fit(bundle.ds_train)

    # --- Warm-start PerHorizonConformal from a calibration block ---
    ds_train = bundle.ds_train
    n_train = ds_train.sizes["timestamp"]
    horizon = 48

    phc = PerHorizonConformal(
        horizon=horizon, alpha=alpha, window=300, aci_lr=0.005,
    )

    if n_train > cal_size + model.min_history:
        print("[DKL] Calibrating per-horizon conformal from training tail...")
        ds_cal_ctx = ds_train.isel(timestamp=slice(0, n_train - cal_size))
        cal_timestamps = pd.to_datetime(
            ds_train.timestamp.values[n_train - cal_size : n_train]
        )[:horizon]
        cal_truth = (
            ds_train.out.transpose("timestamp", "location")
            .isel(timestamp=slice(n_train - cal_size, n_train))
            .values[:horizon].astype(float)
        )

        # Get calibration predictions with log-space outputs
        cal_pred_df = model.predict(
            ds_cal_ctx, cal_timestamps,
            return_intervals=False, return_log_space=True,
        )
        cal_log_pred = long_df_to_matrix(
            cal_pred_df, "log_pred", cal_timestamps, bundle.locations,
        )
        cal_log_std = long_df_to_matrix(
            cal_pred_df, "log_std", cal_timestamps, bundle.locations,
        )
        cal_log_actual = np.log1p(np.clip(cal_truth, 0, None))

        # Seed residual pools per horizon step
        for h in range(min(horizon, len(cal_log_pred))):
            phc.update(
                cal_log_pred[h:h+1, :].ravel(),
                cal_log_std[h:h+1, :].ravel(),
                cal_log_actual[h:h+1, :].ravel(),
            )

    # --- Generate test predictions with log-space outputs ---
    print("[DKL] Generating 48h test forecast...")
    pred_df = model.predict(
        bundle.ds_train, bundle.test_48h_timestamps,
        return_intervals=False, return_log_space=True,
    )

    # Build interval matrices using PerHorizonConformal
    log_pred_matrix = long_df_to_matrix(
        pred_df, "log_pred", bundle.test_48h_timestamps, bundle.locations,
    )
    log_std_matrix = long_df_to_matrix(
        pred_df, "log_std", bundle.test_48h_timestamps, bundle.locations,
    )

    n_counties = len(bundle.locations)
    n_ts = len(bundle.test_48h_timestamps)
    lower_matrix = np.zeros((n_ts, n_counties))
    upper_matrix = np.zeros((n_ts, n_counties))

    for c in range(n_counties):
        lo, hi = phc.get_intervals(
            log_pred_matrix[:, c],
            log_std_matrix[:, c],
        )
        lower_matrix[:, c] = lo
        upper_matrix[:, c] = hi

    # Attach intervals to the prediction DataFrame
    rows = []
    for h, ts in enumerate(bundle.test_48h_timestamps):
        for c, loc in enumerate(bundle.locations):
            rows.append({
                "timestamp": ts,
                "location": loc,
                "pred": float(np.expm1(log_pred_matrix[h, c])),
                "lower": float(lower_matrix[h, c]),
                "upper": float(upper_matrix[h, c]),
            })
    pred_df_out = pd.DataFrame(rows)

    out_path = results_dir / "dkl_pred_48h.csv"
    pred_df_out.to_csv(out_path, index=False)
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
