from pathlib import Path
import numpy as np
import pandas as pd

from utils.dataloader import load_competition_data, temporal_split
from evaluation.metrics import evaluate_all_metrics, long_df_to_matrix, rmse
from models.sarimax import CountySARIMAX
from models.seq2seq import Seq2SeqForecaster


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

    # Save validation predictions
    sarimax_val_pred.to_csv(RESULTS_DIR / "sarimax_val_pred.csv", index=False)
    seq2seq_val_pred.to_csv(RESULTS_DIR / "seq2seq_val_pred.csv", index=False)
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


if __name__ == "__main__":
    main()