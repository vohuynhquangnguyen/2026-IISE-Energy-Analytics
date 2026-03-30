from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CompetitionScores:
    s1_normal_rmse: float
    s2_tail_rmse: float
    s3_f1_nonzero: float
    s4_winkler_95: float


def _as_2d_float(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {arr.shape}")
    return arr


def clip_negative_predictions(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred, dtype=float)
    return np.clip(y_pred, 0.0, None)


def validate_interval_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lo = np.minimum(lower, upper)
    hi = np.maximum(lower, upper)
    return lo, hi


def compute_county_thresholds(train_outages: np.ndarray, q: float = 0.95) -> np.ndarray:
    """
    train_outages: shape (T_train, C)
    Returns:
        tau: shape (C,)
    """
    train_outages = _as_2d_float(train_outages)
    return np.quantile(train_outages, q=q, axis=0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def macro_rmse_by_mask(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    empty_value: float = np.nan,
) -> float:
    """
    Macro-average county-level RMSE over masked entries.

    y_true, y_pred, mask: shape (T, C)
    """
    y_true = _as_2d_float(y_true)
    y_pred = _as_2d_float(y_pred)
    mask = np.asarray(mask, dtype=bool)

    if y_true.shape != y_pred.shape or y_true.shape != mask.shape:
        raise ValueError("y_true, y_pred, and mask must have the same shape.")

    per_county = []
    for c in range(y_true.shape[1]):
        idx = mask[:, c]
        if idx.sum() == 0:
            per_county.append(empty_value)
        else:
            per_county.append(
                np.sqrt(np.mean((y_pred[idx, c] - y_true[idx, c]) ** 2))
            )

    return float(np.nanmean(per_county))


def normal_case_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: np.ndarray,
) -> float:
    """
    s1: macro-averaged RMSE over N_c = {t : y_{c,t} < tau_c}
    """
    y_true = _as_2d_float(y_true)
    y_pred = clip_negative_predictions(_as_2d_float(y_pred))
    tau = np.asarray(tau, dtype=float).reshape(1, -1)

    if tau.shape[1] != y_true.shape[1]:
        raise ValueError("tau must have one threshold per county.")

    mask = y_true < tau
    return macro_rmse_by_mask(y_true, y_pred, mask)


def tail_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tau: np.ndarray,
) -> float:
    """
    s2: macro-averaged RMSE over E_c = {t : y_{c,t} >= tau_c}
    """
    y_true = _as_2d_float(y_true)
    y_pred = clip_negative_predictions(_as_2d_float(y_pred))
    tau = np.asarray(tau, dtype=float).reshape(1, -1)

    if tau.shape[1] != y_true.shape[1]:
        raise ValueError("tau must have one threshold per county.")

    mask = y_true >= tau
    return macro_rmse_by_mask(y_true, y_pred, mask)


def detection_f1_nonzero(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    s3: F1 over all county-hour pairs using
        z_true = 1{y_true >= 1}
        z_pred = 1{y_pred >= 1}
    """
    y_true = _as_2d_float(y_true)
    y_pred = clip_negative_predictions(_as_2d_float(y_pred))

    z_true = (y_true >= 1.0).astype(int).ravel()
    z_pred = (y_pred >= 1.0).astype(int).ravel()

    tp = np.sum((z_true == 1) & (z_pred == 1))
    fp = np.sum((z_true == 0) & (z_pred == 1))
    fn = np.sum((z_true == 1) & (z_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def winkler_score_95(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    s4: average Winkler score over all county-hour pairs.
    Lower is better.
    """
    y_true = _as_2d_float(y_true)
    lower = _as_2d_float(lower)
    upper = _as_2d_float(upper)

    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("y_true, lower, and upper must have the same shape.")

    lower, upper = validate_interval_bounds(lower, upper)

    width = upper - lower
    below_penalty = (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    above_penalty = (2.0 / alpha) * (y_true - upper) * (y_true > upper)

    winkler = width + below_penalty + above_penalty
    return float(np.mean(winkler))


def evaluate_all_metrics(
    train_outages: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> CompetitionScores:
    """
    Convenience wrapper for local validation.
    """
    train_outages = _as_2d_float(train_outages)
    y_true = _as_2d_float(y_true)
    y_pred = clip_negative_predictions(_as_2d_float(y_pred))

    if train_outages.shape[1] != y_true.shape[1]:
        raise ValueError("train_outages and y_true must have the same county dimension.")

    tau = compute_county_thresholds(train_outages, q=0.95)

    s1 = normal_case_rmse(y_true, y_pred, tau)
    s2 = tail_rmse(y_true, y_pred, tau)
    s3 = detection_f1_nonzero(y_true, y_pred)

    if lower is None or upper is None:
        # For point-forecast-only baselines, returning NaN is better than inventing a fake score.
        s4 = np.nan
    else:
        s4 = winkler_score_95(y_true, lower, upper, alpha=alpha)

    return CompetitionScores(
        s1_normal_rmse=s1,
        s2_tail_rmse=s2,
        s3_f1_nonzero=s3,
        s4_winkler_95=s4,
    )


def long_df_to_matrix(
    df: pd.DataFrame,
    value_col: str,
    timestamps: list | np.ndarray | pd.Index,
    locations: list | np.ndarray | pd.Index,
) -> np.ndarray:
    """
    Convert long-format predictions or truth to shape (T, C).
    """
    pivot = (
        df.copy()
        .assign(
            timestamp=pd.to_datetime(df["timestamp"]),
            location=df["location"].astype(str),
        )
        .pivot(index="timestamp", columns="location", values=value_col)
        .reindex(index=pd.to_datetime(timestamps), columns=[str(x) for x in locations])
    )

    return pivot.to_numpy(dtype=float)


def evaluate_from_dataframes(
    train_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    timestamps: list | np.ndarray | pd.Index,
    locations: list | np.ndarray | pd.Index,
    pred_col: str = "pred",
    lower_col: str = "lower",
    upper_col: str = "upper",
    alpha: float = 0.05,
) -> CompetitionScores:
    """
    DataFrame-friendly wrapper.
    Expects:
      - train_df with columns [timestamp, location, out]
      - truth_df with columns [timestamp, location, out]
      - pred_df with columns [timestamp, location, pred, optional lower, upper]
    """
    train_outages = long_df_to_matrix(train_df, "out", sorted(train_df["timestamp"].unique()), locations)
    y_true = long_df_to_matrix(truth_df, "out", timestamps, locations)
    y_pred = long_df_to_matrix(pred_df, pred_col, timestamps, locations)

    lower = upper = None
    if lower_col in pred_df.columns and upper_col in pred_df.columns:
        lower = long_df_to_matrix(pred_df, lower_col, timestamps, locations)
        upper = long_df_to_matrix(pred_df, upper_col, timestamps, locations)

    return evaluate_all_metrics(
        train_outages=train_outages,
        y_true=y_true,
        y_pred=y_pred,
        lower=lower,
        upper=upper,
        alpha=alpha,
    )