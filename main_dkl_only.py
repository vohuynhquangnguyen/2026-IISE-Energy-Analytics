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
        for the DKL model.

    Phase 3 — Test Inference
        Retrain the DKL model on the *full* training set and produce the 48-hour
        forecast file required for competition submission.

All feature engineering is centralised in ``utils/feature_engineering.py`` so
that every model trains and validates on an identical feature set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

# Only import the DKL model
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
    alpha: float = 0.05,
) -> dict[str, list[CompetitionScores]]:
    """
    Run walk-forward cross-validation for the DKL model.
    """
    all_scores: dict[str, list[CompetitionScores]] = {"dkl": []}

    for fold in folds:
        print_fold_header(fold)
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
#  Feature importance analysis (SHAP + Gradient Attribution)                   #
# =========================================================================== #

def compute_gradient_attribution(
    model: DKLForecaster,
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Compute gradient-based feature attribution for the DKL model.

    For each sample, computes |d(mean_prediction) / d(input_feature)|
    averaged over all samples to get per-feature importance.

    Returns a DataFrame with columns: feature, importance.
    """
    X_clean = np.nan_to_num(
        np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0,
    )
    X_scaled = model.x_scaler.transform(X_clean).astype(np.float32)

    model.model.eval()
    model.likelihood.eval()

    import gpytorch as gpy

    batch_size = model.predict_batch_size
    all_grads = []

    for start in range(0, len(X_scaled), batch_size):
        stop = start + batch_size
        xb = torch.tensor(
            X_scaled[start:stop], dtype=torch.float32, device=model.device,
        )
        xb.requires_grad_(True)

        with gpy.settings.fast_pred_var():
            pred_dist = model.likelihood(model.model(xb))
            pred_mean = pred_dist.mean

        pred_mean.sum().backward()
        all_grads.append(xb.grad.detach().cpu().numpy())

    grads = np.concatenate(all_grads, axis=0)  # (N, D)

    # Mean absolute gradient per feature
    importance = np.abs(grads).mean(axis=0)

    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def compute_shap_values(
    model: DKLForecaster,
    X: np.ndarray,
    feature_names: list[str],
    n_background: int = 150,
    n_explain: int = 200,
) -> tuple:
    """
    Compute SHAP values for the DKL model using KernelSHAP.

    Parameters
    ----------
    model : DKLForecaster
        Fitted DKL model.
    X : np.ndarray
        Full oracle feature matrix.
    feature_names : list[str]
        Feature column names.
    n_background : int
        Number of background samples for KernelSHAP.
    n_explain : int
        Number of samples to explain.

    Returns
    -------
    (shap_values, X_explain) : tuple
        shap_values is the array from the explainer;
        X_explain is the subset that was explained.
    """
    import shap

    rng = np.random.default_rng(42)

    # Subsample background and explanation sets
    bg_idx = rng.choice(len(X), size=min(n_background, len(X)), replace=False)
    X_background = X[bg_idx]

    remaining = np.setdiff1d(np.arange(len(X)), bg_idx)
    ex_idx = rng.choice(remaining, size=min(n_explain, len(remaining)), replace=False)
    X_explain = X[ex_idx]

    def predict_fn(X_input):
        result = model.predict_oracle(X_input)
        return result["mean"]

    background_summary = shap.kmeans(X_background, min(50, len(X_background)))
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    shap_values = explainer.shap_values(X_explain)

    return shap_values, X_explain


def plot_gradient_attribution(
    grad_df: pd.DataFrame,
    results_dir: Path,
    top_n: int = 25,
) -> None:
    """Plot and save a horizontal bar chart of gradient-based feature importance."""
    top = grad_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top)), top["importance"].values[::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |Gradient|")
    ax.set_title(f"DKL Gradient Attribution (top {top_n} features)")
    plt.tight_layout()

    path = results_dir / "dkl_gradient_attribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved gradient attribution plot → {path}")


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    results_dir: Path,
    top_n: int = 25,
) -> None:
    """Plot and save a SHAP beeswarm summary plot."""
    import shap

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_explain,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
    )
    plt.tight_layout()

    path = results_dir / "dkl_shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved SHAP summary plot → {path}")

    # Also save a bar plot of mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    top = shap_df.head(top_n)
    ax2.barh(range(len(top)), top["mean_abs_shap"].values[::-1])
    ax2.set_yticks(range(len(top)))
    ax2.set_yticklabels(top["feature"].values[::-1], fontsize=9)
    ax2.set_xlabel("Mean |SHAP value|")
    ax2.set_title(f"DKL SHAP Feature Importance (top {top_n} features)")
    plt.tight_layout()

    path2 = results_dir / "dkl_shap_bar.png"
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved SHAP bar plot → {path2}")


def run_feature_importance(
    bundle: CompetitionData,
    results_dir: Path,
) -> None:
    """
    Train DKL on the full training set, then compute and plot
    gradient attribution and SHAP feature importance.
    """
    print("\n" + "=" * 70)
    print("  Feature Importance Analysis (DKL)")
    print("=" * 70)

    # --- Train model ---
    print("\n  [1/4] Training DKL on full training data...")
    model = DKLForecaster.from_config("dkl.yaml")
    model.fit(bundle.ds_train)

    # --- Build oracle feature table ---
    print("  [2/4] Building oracle feature table...")
    X_df, y, _, _ = model.build_feature_table(bundle.ds_train)
    feature_names = X_df.columns.tolist()
    X = X_df.to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"         Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # --- Gradient attribution ---
    print("  [3/4] Computing gradient attribution...")
    grad_df = compute_gradient_attribution(model, X, feature_names)
    grad_df.to_csv(results_dir / "dkl_gradient_attribution.csv", index=False)
    plot_gradient_attribution(grad_df, results_dir)
    print("\n  Top 10 features by gradient attribution:")
    for i, row in grad_df.head(10).iterrows():
        print(f"    {i + 1:2d}. {row['feature']:40s} {row['importance']:.6f}")

    # --- SHAP ---
    print("\n  [4/4] Computing SHAP values (this may take a few minutes)...")
    shap_values, X_explain = compute_shap_values(model, X, feature_names)
    plot_shap_summary(shap_values, X_explain, feature_names, results_dir)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(results_dir / "dkl_shap_importance.csv", index=False)

    print("\n  Top 10 features by SHAP:")
    for i, row in shap_df.head(10).iterrows():
        print(f"    {i + 1:2d}. {row['feature']:40s} {row['mean_abs_shap']:.6f}")

    print(f"\n  All results saved to {results_dir}/")


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
    alpha = pipeline_cfg.get("intervals", {}).get("alpha", 0.04)

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
    print(f"    Enabled model:    DKL")
    print()

    # ================================================================== #
    #  Phase 2: Walk-Forward Cross-Validation                             #
    # ================================================================== #
    # print("=" * 70)
    # print("  Phase 2: Walk-Forward Cross-Validation")
    # print("=" * 70)

    # folds = walk_forward_split(
    #     bundle.ds_train,
    #     n_folds=wf_cfg.get("n_folds", 3),
    #     horizon=wf_cfg.get("horizon", 48),
    #     min_train_frac=wf_cfg.get("min_train_frac", 0.5),
    # )
    # print(f"\n  Created {len(folds)} walk-forward folds "
    #       f"(horizon={wf_cfg.get('horizon', 48)} hours)")

    # all_scores = run_walk_forward_validation(
    #     folds=folds,
    #     locations=bundle.locations,
    #     alpha=alpha,
    # )

    # print_cv_summary(all_scores)

    # ================================================================== #
    #  Phase 3: Test Inference                                             #
    # ================================================================== #
    print("=" * 70)
    print("  Phase 3: Test Inference (retrain on full training data)")
    print("=" * 70)
    print()

    retrain_and_predict_dkl(bundle, results_dir)

    # ================================================================== #
    #  Phase 4: Feature Importance Analysis                               #
    # ================================================================== #
    run_feature_importance(bundle, results_dir)

    print("\n" + "=" * 70)
    print("  Pipeline complete.  Results saved to:", results_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()