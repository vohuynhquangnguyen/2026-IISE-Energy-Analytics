from pathlib import Path
import numpy as np
import pandas as pd
import torch

from utils.dataloader import load_competition_data, temporal_split
from evaluation.metrics import evaluate_all_metrics, long_df_to_matrix, rmse
from models.sarimax import CountySARIMAX
from models.seq2seq import Seq2SeqForecaster
from models.dkl import DKLForecaster


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SEQ2SEQ_CONFIG = {
    "seq_len": 24,
    "horizon": 48,
    "hidden_dim": 64,
    "num_layers": 1,
    "batch_size": 64,
    "epochs": 5,
    "lr": 1e-3,
}

DKL_CONFIG = {
    "lags": (1, 2, 3, 4, 5, 6, 12, 24, 48),
    "rolling_windows": (6, 12, 24),
    "hidden_dims": (256, 128, 64),
    "embed_dim": 20,
    "num_inducing": 768,
    "dropout": 0.1,
    "epochs": 40,
    "batch_size": 1024,
    "lr": 5e-3,
    "max_train_rows": None,
    "random_state": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "clip_nonnegative": True,
    "num_workers": 0,
    "predict_batch_size": 4096,
    "grad_clip_norm": 1.0,
    "weather_lags": (1, 3, 6, 12, 24),
    "weather_rolling_windows": (6, 12, 24),
    "lag_noise_frac": 0.3,
    "lag_noise_scale": 0.15,
}

def average_county_rmse(y_true_matrix, pred_df, locations, timestamps):
    """
    Demo-notebook style metric:
    macro average of county-level RMSE over the validation horizon.
    """
    pred_matrix = long_df_to_matrix(
        pred_df,
        value_col="pred",
        timestamps=timestamps,
        locations=locations,
    )

    county_rmses = []
    for i in range(len(locations)):
        county_rmses.append(rmse(y_true_matrix[:, i], pred_matrix[:, i]))
    return float(np.nanmean(county_rmses)), pred_matrix


def print_scores(model_name, demo_rmse, comp_scores):
    print(f"\n[{model_name}] Validation results")
    print(f"  Avg county RMSE (demo style): {demo_rmse:.4f}")
    print(f"  Normal-case RMSE:             {comp_scores.s1_normal_rmse:.4f}")
    print(f"  Tail RMSE:                    {comp_scores.s2_tail_rmse:.4f}")
    print(f"  Nonzero-outage F1:            {comp_scores.s3_f1_nonzero:.4f}")
    if np.isnan(comp_scores.s4_winkler_95):
        print("  Winkler 95:                   NaN (no lower/upper intervals yet)")
    else:
        print(f"  Winkler 95:                   {comp_scores.s4_winkler_95:.4f}")


def split_for_calibration(ds, cal_size=48):
    """
    Split a historical dataset into:
      - fit portion
      - trailing calibration portion

    The calibration block is used only to estimate interval widths.
    """
    n = ds.sizes["timestamp"]
    if n <= cal_size:
        raise ValueError(
            f"Need more than cal_size={cal_size} timestamps, got n={n}."
        )

    ds_fit = ds.isel(timestamp=slice(0, n - cal_size))
    ds_cal = ds.isel(timestamp=slice(n - cal_size, n))
    return ds_fit, ds_cal


def add_quantile_intervals(pred_df, q_by_county, locations):
    """
    Add lower/upper columns to a long prediction DataFrame using
    county-specific absolute-residual quantiles.
    """
    q_map = {str(loc): float(q_by_county[i]) for i, loc in enumerate(locations)}

    out = pred_df.copy()
    out["location"] = out["location"].astype(str)
    out["q"] = out["location"].map(q_map).astype(float)
    out["lower"] = np.clip(out["pred"] - out["q"], 0, None)
    out["upper"] = out["pred"] + out["q"]
    out = out.drop(columns=["q"])
    return out


def estimate_seq2seq_interval_widths(ds, locations, model_config, q_level=0.95, cal_size=48):
    """
    Estimate county-specific interval half-widths for Seq2Seq
    using a simple conformal-style calibration block.

    Procedure:
      1. hold out the last cal_size timestamps of ds as calibration
      2. train Seq2Seq on the earlier portion
      3. forecast the calibration horizon
      4. compute county-wise q_level quantile of absolute residuals
    """
    ds_fit, ds_cal = split_for_calibration(ds, cal_size=cal_size)

    cal_timestamps = pd.to_datetime(ds_cal.timestamp.values[: model_config["horizon"]])
    cal_truth = (
        ds_cal.out.transpose("timestamp", "location")
        .isel(timestamp=slice(0, model_config["horizon"]))
        .values.astype(float)
    )

    calib_model = Seq2SeqForecaster(**model_config)
    calib_model.fit(ds_fit)

    cal_pred_df = calib_model.predict(ds_fit, cal_timestamps)
    cal_pred = long_df_to_matrix(
        cal_pred_df,
        value_col="pred",
        timestamps=cal_timestamps,
        locations=locations,
    )

    abs_resid = np.abs(cal_truth - cal_pred)  # shape (48, C)
    q_by_county = np.quantile(abs_resid, q_level, axis=0)

    return q_by_county


def validate_sarimax(split, locations):
    print("Training SARIMAX...")
    model = CountySARIMAX(order=(1, 0, 1))
    model.fit(split.ds_train_sub)

    pred_df = model.predict(
        split.val_timestamps_48h,
        locations,
        return_intervals=True,
        alpha=0.05,
    )

    demo_rmse, pred_matrix = average_county_rmse(
        split.val_truth_48h,
        pred_df,
        locations,
        split.val_timestamps_48h,
    )

    train_outages = (
        split.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    lower_matrix = long_df_to_matrix(
        pred_df, "lower", split.val_timestamps_48h, locations
    )
    upper_matrix = long_df_to_matrix(
        pred_df, "upper", split.val_timestamps_48h, locations
    )

    comp_scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=split.val_truth_48h,
        y_pred=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
    )

    print_scores("SARIMAX", demo_rmse, comp_scores)
    return model, pred_df, comp_scores


def validate_seq2seq(split, locations):
    print("Estimating Seq2Seq interval widths from calibration block...")
    q_by_county = estimate_seq2seq_interval_widths(
        ds=split.ds_train_sub,
        locations=locations,
        model_config=SEQ2SEQ_CONFIG,
        q_level=0.95,
        cal_size=48,
    )

    print("Training Seq2Seq...")
    model = Seq2SeqForecaster(**SEQ2SEQ_CONFIG)
    model.fit(split.ds_train_sub)

    # Use ds_train_sub as the historical context,
    # matching the history-only forecasting logic from demo.ipynb.
    pred_df = model.predict(split.ds_train_sub, split.val_timestamps_48h)
    pred_df = add_quantile_intervals(pred_df, q_by_county, locations)

    demo_rmse, pred_matrix = average_county_rmse(
        split.val_truth_48h,
        pred_df,
        locations,
        split.val_timestamps_48h,
    )

    train_outages = (
        split.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    lower_matrix = long_df_to_matrix(
        pred_df, "lower", split.val_timestamps_48h, locations
    )
    upper_matrix = long_df_to_matrix(
        pred_df, "upper", split.val_timestamps_48h, locations
    )

    comp_scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=split.val_truth_48h,
        y_pred=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
    )

    print_scores("Seq2Seq", demo_rmse, comp_scores)
    return model, pred_df, comp_scores, q_by_county


def save_demo_predictions(
    bundle,
    model_name,
    model,
    is_seq2seq=False,
    seq2seq_q_by_county=None,
):
    """
    Retrain-on-full-train style prediction step, similar to demo.ipynb,
    but saving interval columns too.
    """
    if bundle.ds_test_48h is None:
        print(f"[{model_name}] No test_48h_demo.nc found. Skipping demo test prediction.")
        return

    test_timestamps = bundle.test_48h_timestamps
    locations = bundle.locations

    if is_seq2seq:
        pred_df = model.predict(bundle.ds_train, test_timestamps)

        if seq2seq_q_by_county is None:
            raise ValueError("seq2seq_q_by_county must be provided for Seq2Seq interval output.")

        pred_df = add_quantile_intervals(pred_df, seq2seq_q_by_county, locations)
    else:
        pred_df = model.predict(
            test_timestamps,
            locations,
            return_intervals=True,
            alpha=0.05,
        )

    out_path = RESULTS_DIR / f"{model_name.lower()}_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[{model_name}] Saved demo predictions to {out_path}")


def validate_dkl(split, locations, interval_method="conformal"):
    print("Training DKL...")
    model = DKLForecaster(**DKL_CONFIG)
    model.fit(split.ds_train_sub)

    # Get point predictions first (no intervals)
    pred_df = model.predict(
        split.ds_train_sub,
        split.val_timestamps_48h,
        return_intervals=False,
    )

    demo_rmse, pred_matrix = average_county_rmse(
        split.val_truth_48h,
        pred_df,
        locations,
        split.val_timestamps_48h,
    )

    # Apply adaptive conformal intervals on the point predictions
    y_true_flat = split.val_truth_48h.ravel()
    pred_flat   = pred_matrix.ravel()
    alpha = 0.05

    residuals = np.abs(y_true_flat - pred_flat)
    n_cal = len(residuals)
    conformal_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    conformal_q = np.quantile(residuals, conformal_level)

    # Fixed conformal
    conf_lo = np.clip(pred_flat - conformal_q, 0, None).reshape(pred_matrix.shape)
    conf_hi = (pred_flat + conformal_q).reshape(pred_matrix.shape)

    # Adaptive conformal
    median_nz = np.median(pred_flat[pred_flat > 0]) if (pred_flat > 0).any() else 1.0
    adapt_q = np.where(pred_flat > median_nz, conformal_q * 1.5, conformal_q * 0.8)
    adapt_q = np.maximum(adapt_q, 5.0)
    adapt_lo = np.clip(pred_flat - adapt_q, 0, None).reshape(pred_matrix.shape)
    adapt_hi = (pred_flat + adapt_q).reshape(pred_matrix.shape)

    # Pick best
    conf_cov  = np.mean((y_true_flat >= conf_lo.ravel()) & (y_true_flat <= conf_hi.ravel()))
    adapt_cov = np.mean((y_true_flat >= adapt_lo.ravel()) & (y_true_flat <= adapt_hi.ravel()))

    def _wk(lo, hi, yt):
        return np.mean((hi-lo) + (2/alpha)*np.maximum(lo-yt,0) + (2/alpha)*np.maximum(yt-hi,0))

    conf_wk  = _wk(conf_lo.ravel(), conf_hi.ravel(), y_true_flat)
    adapt_wk = _wk(adapt_lo.ravel(), adapt_hi.ravel(), y_true_flat)

    print(f"\n  PI strategies: Fixed(cov={conf_cov:.3f}, wk={conf_wk:.1f}) "
          f"| Adaptive(cov={adapt_cov:.3f}, wk={adapt_wk:.1f})")

    if adapt_cov >= 0.94 and adapt_wk < conf_wk:
        lower_matrix, upper_matrix = adapt_lo, adapt_hi
        print(f"  → Using: Adaptive Conformal")
    else:
        lower_matrix, upper_matrix = conf_lo, conf_hi
        print(f"  → Using: Conformal (fixed)")

    train_outages = (
        split.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    comp_scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=split.val_truth_48h,
        y_pred=pred_matrix,
        lower=lower_matrix,
        upper=upper_matrix,
    )

    print_scores("DKL", demo_rmse, comp_scores)
    return model, pred_df, comp_scores


def validate_dkl_oracle(bundle, split, locations):
    """
    Oracle evaluation — replicates the colleague's notebook cells 14-19.

    Key differences from the autoregressive validate_dkl:
    - Features built from ground-truth (no autoregressive loop)
    - Split: last 48h for val, everything before for train (~97.8% train)
    - No noise injection (fair comparison to colleague)
    - Adaptive conformal intervals
    """
    from sklearn.metrics import f1_score as sk_f1_score

    print("\n" + "=" * 70)
    print("DKL ORACLE EVALUATION  (colleague's methodology)")
    print("=" * 70)

    # 1) Build features from full dataset using actual ground-truth values
    print("[DKL-oracle] Building feature table...")
    oracle_config = dict(DKL_CONFIG)
    oracle_config["lag_noise_frac"] = 0.0  # No noise for fair comparison
    model = DKLForecaster(**oracle_config)
    model.locations_ = [str(loc) for loc in bundle.ds_train.location.values]
    X_df, y, row_ts, row_locs = model.build_feature_table(bundle.ds_train)
    model.feature_columns_ = X_df.columns.tolist()
    X_all = X_df.values.astype(np.float32)
    print(f"  Total: {len(y):,} samples, {X_all.shape[1]} features")

    # 2) Split: last 48 timestamps for val (matching colleague's cell 14)
    all_timestamps = pd.to_datetime(bundle.ds_train.timestamp.values)
    val_start = all_timestamps[-48]
    val_end   = all_timestamps[-1]

    train_mask = row_ts < val_start
    val_mask   = (row_ts >= val_start) & (row_ts <= val_end)

    X_train, y_train = X_all[train_mask], y[train_mask]
    X_val,   y_val   = X_all[val_mask],   y[val_mask]
    val_locs_arr     = row_locs[val_mask]

    # Drop rows with NaN in lag-48 columns (same as colleague's cell 14)
    lag48_idx = [i for i, c in enumerate(model.feature_columns_) if "lag_48" in c]
    if lag48_idx:
        tv = ~np.isnan(X_train[:, lag48_idx]).any(axis=1)
        vv = ~np.isnan(X_val[:, lag48_idx]).any(axis=1)
        X_train, y_train = X_train[tv], y_train[tv]
        X_val, y_val     = X_val[vv], y_val[vv]
        val_locs_arr     = val_locs_arr[vv]

    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}")
    print(f"  Val period: {val_start} to {val_end}")

    # 3) Train (no noise injection)
    model.fit_from_arrays(X_train, y_train, val_X=X_val, val_y=y_val)

    # 4) Get point predictions
    pred_val = model.predict_oracle(X_val, ci_multiplier=2.0)["mean"]

    # ================================================================ #
    #  CONFORMAL + ADAPTIVE CONFORMAL  (colleague's cell 18)           #
    # ================================================================ #
    alpha = 0.05
    residuals = np.abs(y_val - pred_val)

    # Fixed conformal
    n_cal = len(residuals)
    conformal_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    conformal_q = np.quantile(residuals, conformal_level)

    conf_lo = np.clip(pred_val - conformal_q, 0, None)
    conf_hi = pred_val + conformal_q
    conf_cov = np.mean((y_val >= conf_lo) & (y_val <= conf_hi))
    conf_wk  = np.mean(
        (conf_hi - conf_lo)
        + (2/alpha) * np.maximum(conf_lo - y_val, 0)
        + (2/alpha) * np.maximum(y_val - conf_hi, 0)
    )

    # Adaptive conformal (colleague's approach)
    median_nz = np.median(pred_val[pred_val > 0]) if (pred_val > 0).any() else 1.0
    adapt_q = np.where(
        pred_val > median_nz,
        conformal_q * 1.5,
        conformal_q * 0.8,
    )
    adapt_q = np.maximum(adapt_q, 5.0)

    adapt_lo = np.clip(pred_val - adapt_q, 0, None)
    adapt_hi = pred_val + adapt_q
    adapt_cov = np.mean((y_val >= adapt_lo) & (y_val <= adapt_hi))
    adapt_wk  = np.mean(
        (adapt_hi - adapt_lo)
        + (2/alpha) * np.maximum(adapt_lo - y_val, 0)
        + (2/alpha) * np.maximum(y_val - adapt_hi, 0)
    )

    # Choose best strategy
    print("\n=== PI Strategy Comparison ===")
    strategies = {
        "Conformal (fixed)":    (conf_lo,  conf_hi,  conf_cov,  conf_wk),
        "Adaptive Conformal":   (adapt_lo, adapt_hi, adapt_cov, adapt_wk),
    }
    best_name, best_wk = None, float("inf")
    for name, (lo, hi, cov, wk) in strategies.items():
        tag = "✓" if cov >= 0.94 else "✗"
        print(f"  {tag} {name}: coverage={cov:.4f}, Winkler={wk:.2f}")
        if cov >= 0.94 and wk < best_wk:
            best_wk = wk
            best_name = name

    if best_name is None:
        best_name = "Conformal (fixed)"
    print(f"  → Using: {best_name}")

    lower, upper = strategies[best_name][0], strategies[best_name][1]
    pred = pred_val

    # Enforce constraints
    upper = np.maximum(upper, lower + 1)
    lower = np.minimum(lower, pred)
    upper = np.maximum(upper, pred)

    # 5) Compute competition metrics — threshold from ALL data before val
    train_out_all = (
        bundle.ds_train.out.transpose("timestamp", "location")
        .isel(timestamp=slice(0, len(all_timestamps) - 48))
        .values.astype(float)
    )
    tau = np.quantile(train_out_all, 0.95, axis=0)
    loc_idx = {str(loc): i for i, loc in enumerate(locations)}

    s1_list, s2_list = [], []
    for loc in locations:
        m = val_locs_arr == loc
        if m.sum() == 0:
            continue
        yt, yp = y_val[m], pred[m]
        c = loc_idx[loc]
        norm = yt < tau[c]
        if norm.sum() > 0:
            s1_list.append(np.sqrt(np.mean((yt[norm] - yp[norm])**2)))
        ext = yt >= tau[c]
        if ext.sum() > 0:
            s2_list.append(np.sqrt(np.mean((yt[ext] - yp[ext])**2)))

    z_true = (y_val >= 1).astype(int)
    z_pred = (pred >= 1).astype(int)
    s3 = sk_f1_score(z_true, z_pred)
    best_f1, best_t = s3, 1.0
    for t in np.arange(0.1, 3.0, 0.1):
        f = sk_f1_score(z_true, (pred >= t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, t

    winkler = (upper - lower) \
        + (2/alpha)*np.maximum(lower - y_val, 0) \
        + (2/alpha)*np.maximum(y_val - upper, 0)
    coverage = np.mean((y_val >= lower) & (y_val <= upper))

    print("\n" + "=" * 60)
    print("DKL ORACLE — VALIDATION RESULTS")
    print(f"(PI method: {best_name})")
    print("=" * 60)
    print(f"s1 (Normal RMSE):   {np.mean(s1_list):.4f}")
    print(f"s2 (Extreme RMSE):  {np.mean(s2_list):.4f}")
    print(f"s3 (F1 Detection):  {best_f1:.4f}")
    print(f"s4 (Winkler Score): {np.mean(winkler):.4f}")
    print(f"PI Coverage:        {coverage:.4f}")
    print(f"Mean PI width:      {np.mean(upper - lower):.2f}")
    print(f"Overall RMSE:       {np.sqrt(np.mean((y_val - pred)**2)):.4f}")
    print(f"Conformal quantile: {conformal_q:.2f}")

    return model, {"mean": pred, "lower": lower, "upper": upper,
                    "conformal_q": conformal_q, "pi_method": best_name}


def save_dkl_demo_predictions(bundle, interval_method="conformal"):
    if bundle.ds_test_48h is None:
        print("[DKL] No test_48h_demo.nc found. Skipping demo test prediction.")
        return

    print("[DKL] Retraining on full training data...")
    model = DKLForecaster(**DKL_CONFIG)
    model.fit(bundle.ds_train)

    if interval_method == "conformal":
        print("[DKL] Calibrating conformal intervals on full training data...")
        model.calibrate_intervals(
            bundle.ds_train,
            calibration_size=48,
            alpha=0.05,
        )

    pred_df = model.predict(
        bundle.ds_train,
        bundle.test_48h_timestamps,
        return_intervals=True,
        alpha=0.05,
        interval_method=interval_method,
    )

    out_path = RESULTS_DIR / "dkl_pred_48h.csv"
    pred_df.to_csv(out_path, index=False)
    print(f"[DKL] Saved demo predictions to {out_path}")


def main():
    bundle = load_competition_data("data")
    split = temporal_split(bundle.ds_train, val_fraction=0.2)

    print("Dataset summary")
    print(f"  Train timestamps:   {bundle.ds_train.sizes['timestamp']}")
    print(f"  Counties:           {bundle.ds_train.sizes['location']}")
    print(f"  Weather features:   {bundle.ds_train.sizes['feature']}")
    print(f"  Train split size:   {split.ds_train_sub.sizes['timestamp']}")
    print(f"  Validation size:    {split.ds_val.sizes['timestamp']}")
    print(f"  Validation horizon: {len(split.val_timestamps_48h)}")
    print()

    locations = bundle.locations

    # Validation phase
    sarimax_model, sarimax_val_pred, sarimax_scores = validate_sarimax(split, locations)
    seq2seq_model, seq2seq_val_pred, seq2seq_scores, _ = validate_seq2seq(split, locations)
    dkl_model, dkl_val_pred, dkl_scores = validate_dkl(split, locations, interval_method="conformal")

    # Oracle evaluation with adaptive conformal — comparable to colleague's notebook
    dkl_oracle_model, dkl_oracle_results = validate_dkl_oracle(
        bundle, split, locations
    )

    # Save validation predictions
    sarimax_val_pred.to_csv(RESULTS_DIR / "sarimax_val_pred.csv", index=False)
    seq2seq_val_pred.to_csv(RESULTS_DIR / "seq2seq_val_pred.csv", index=False)
    dkl_val_pred.to_csv(RESULTS_DIR / "dkl_val_pred.csv", index=False)
    print("\nSaved validation prediction files to results/")

    # Retrain on full train set for demo test inference
    print("\nRetraining models on full training data for demo test prediction...")

    sarimax_full = CountySARIMAX(order=(1, 0, 1))
    sarimax_full.fit(bundle.ds_train)
    save_demo_predictions(bundle, "SARIMAX", sarimax_full, is_seq2seq=False)

    # Estimate Seq2Seq interval widths from a calibration split on full training data
    seq2seq_q_full = estimate_seq2seq_interval_widths(
        ds=bundle.ds_train,
        locations=locations,
        model_config=SEQ2SEQ_CONFIG,
        q_level=0.95,
        cal_size=48,
    )

    seq2seq_full = Seq2SeqForecaster(**SEQ2SEQ_CONFIG)
    seq2seq_full.fit(bundle.ds_train)
    save_demo_predictions(
        bundle,
        "Seq2Seq",
        seq2seq_full,
        is_seq2seq=True,
        seq2seq_q_by_county=seq2seq_q_full,
    )

    # DKL demo test predictions
    save_dkl_demo_predictions(bundle, interval_method="conformal")

if __name__ == "__main__":
    main()
