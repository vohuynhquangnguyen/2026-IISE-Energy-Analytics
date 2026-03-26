from pathlib import Path
import numpy as np
import pandas as pd

from utils.dataloader import load_competition_data, temporal_split
from evaluation.metrics import evaluate_all_metrics, long_df_to_matrix, rmse
from models.sarimax import CountySARIMAX
from models.seq2seq import Seq2SeqForecaster


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def average_county_rmse(y_true_matrix, pred_df, locations):
    """
    Demo-notebook style metric:
    macro average of county-level RMSE over the validation horizon.
    """
    pred_matrix = long_df_to_matrix(
        pred_df,
        value_col="pred",
        timestamps=pred_df["timestamp"].drop_duplicates().sort_values(),
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
        print(f"  Winkler 95:                   NaN (no lower/upper intervals yet)")
    else:
        print(f"  Winkler 95:                   {comp_scores.s4_winkler_95:.4f}")


def validate_sarimax(split, locations):
    print("Training SARIMAX...")
    model = CountySARIMAX(order=(1, 0, 1))
    model.fit(split.ds_train_sub)

    pred_df = model.predict(split.val_timestamps_48h, locations, return_intervals=True, alpha=0.05)

    demo_rmse, pred_matrix = average_county_rmse(
        split.val_truth_48h,
        pred_df,
        locations,
    )

    train_outages = (
        split.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    pred_matrix = long_df_to_matrix(pred_df, "pred", split.val_timestamps_48h, locations)
    lower_matrix = long_df_to_matrix(pred_df, "lower", split.val_timestamps_48h, locations)
    upper_matrix = long_df_to_matrix(pred_df, "upper", split.val_timestamps_48h, locations)

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
    print("Training Seq2Seq...")
    model = Seq2SeqForecaster(
        seq_len=24,
        horizon=48,
        hidden_dim=64,
        num_layers=1,
        batch_size=64,
        epochs=5,
        lr=1e-3,
    )
    model.fit(split.ds_train_sub)

    # Important: use ds_train_sub as the historical context,
    # exactly like the demo notebook’s history-only forecasting setup.
    pred_df = model.predict(split.ds_train_sub, split.val_timestamps_48h)

    demo_rmse, pred_matrix = average_county_rmse(
        split.val_truth_48h,
        pred_df,
        locations,
    )

    train_outages = (
        split.ds_train_sub.out.transpose("timestamp", "location")
        .values.astype(float)
    )

    abs_resid = np.abs(cal_truth - cal_pred)   # shape (48, C)
    q = np.quantile(abs_resid, 0.95, axis=0)   # shape (C,)
    lower = np.clip(pred_matrix - q.reshape(1, -1), 0, None)
    upper = pred_matrix + q.reshape(1, -1)

    comp_scores = evaluate_all_metrics(
        train_outages=train_outages,
        y_true=split.val_truth_48h,
        y_pred=pred_matrix,
        lower=lower,
        upper=upper,
    )

    print_scores("Seq2Seq", demo_rmse, comp_scores)
    return model, pred_df, comp_scores


def save_demo_predictions(bundle, model_name, model, is_seq2seq=False):
    """
    Retrain-on-full-train style prediction step, similar to demo.ipynb.
    """
    if bundle.ds_test_48h is None:
        print(f"[{model_name}] No test_48h_demo.nc found. Skipping demo test prediction.")
        return

    test_timestamps = bundle.test_48h_timestamps
    locations = bundle.locations

    if is_seq2seq:
        pred_df = model.predict(bundle.ds_train, test_timestamps)
    else:
        pred_df = model.predict(test_timestamps, locations)

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
    seq2seq_model, seq2seq_val_pred, seq2seq_scores = validate_seq2seq(split, locations)

    # Save validation predictions
    sarimax_val_pred.to_csv(RESULTS_DIR / "sarimax_val_pred.csv", index=False)
    seq2seq_val_pred.to_csv(RESULTS_DIR / "seq2seq_val_pred.csv", index=False)
    print("\nSaved validation prediction files to results/")

    # Retrain on full train set for demo test inference
    print("\nRetraining models on full training data for demo test prediction...")

    sarimax_full = CountySARIMAX(order=(1, 0, 1))
    sarimax_full.fit(bundle.ds_train)
    save_demo_predictions(bundle, "SARIMAX", sarimax_full, is_seq2seq=False)

    seq2seq_full = Seq2SeqForecaster(
        seq_len=24,
        horizon=48,
        hidden_dim=64,
        num_layers=1,
        batch_size=64,
        epochs=5,
        lr=1e-3,
    )
    seq2seq_full.fit(bundle.ds_train)
    save_demo_predictions(bundle, "Seq2Seq", seq2seq_full, is_seq2seq=True)


if __name__ == "__main__":
    main()