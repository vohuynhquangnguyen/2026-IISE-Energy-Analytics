"""
utils/feature_engineering.py
============================
Centralised feature-engineering module shared by **all** models so that every
model trains and validates on an identical feature set — guaranteeing
apple-to-apple comparisons.

The module is split into three logical sections:

1. **Tabular features** (used by DKL and any future tree/GP model):
   - Temporal encodings  (hour-of-day, day-of-week, month — cyclic + raw)
   - Outage lag features (configurable lag set)
   - Outage rolling statistics (mean, max, std, EMA)
   - Outage momentum / acceleration / zero-streak / percentile
   - Raw + lagged + rolling + diff weather features
   - Derived weather features (wind speed, temperature extremes, severity)
   - Interaction terms  (weather × outage, time × weather)

2. **Sequence features** (used by Seq2Seq):
   - Z-normalisation helpers
   - Sliding-window construction for encoder–decoder training

3. **SARIMAX exogenous features** (used by CountySARIMAX):
   - Weather regressors extracted from the dataset's weather variable
   - Deterministic temporal encodings (hour, day-of-week, month — cyclic)
   - Wind speed derived from u10/v10 components
   - All features are time-aligned arrays that are available for both the
     historical (training) period and the future (prediction) period

Reference
---------
- demo.ipynb cells 24-25  (Seq2Seq data prep)
- colleague's DKL notebook (tabular feature table)
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# =========================================================================== #
#  Default feature-engineering constants                                       #
#  (can be overridden via YAML config → model constructors)                    #
# =========================================================================== #

# Top-20 weather variables selected for raw features (storm indicators,
# soil/boundary-layer, radiation, precipitation).
DEFAULT_KEY_WEATHER: list[str] = [
    "gust", "cape", "cape_1", "tp", "prate", "refc", "hail",
    "u10", "v10", "t2m", "mstav", "pwat", "sh2", "lftx",
    "blh", "sdlwrf", "sp", "tcc", "r2", "pcdb",
]

# Subset of weather variables for rolling aggregations (mean / max).
DEFAULT_ROLLING_WEATHER: list[str] = [
    "gust", "cape", "tp", "prate", "t2m", "mstav",
]

# Subset of weather variables for first-difference features.
DEFAULT_DIFF_WEATHER: list[str] = [
    "gust", "cape", "t2m", "sp", "prate",
]


# =========================================================================== #
#  1. Tabular feature builder (DKL and future models)                          #
# =========================================================================== #

def _sin_cos(value: int | float, period: int | float) -> tuple[float, float]:
    """Return (sin, cos) encoding for a cyclic variable."""
    angle = 2.0 * math.pi * float(value) / float(period)
    return math.sin(angle), math.cos(angle)


def build_tabular_feature_row(
    out_hist: np.ndarray,
    tracked_hist: np.ndarray,
    weather_hist: np.ndarray,
    target_ts: pd.Timestamp,
    weather_name_to_idx: dict[str, int],
    weather_features: list[str],
    *,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 12, 24, 48),
    rolling_windows: tuple[int, ...] = (6, 12, 24),
    weather_subset: list[str] | None = None,
    rolling_weather_subset: list[str] | None = None,
    diff_weather_subset: list[str] | None = None,
    weather_lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    weather_rolling_windows: tuple[int, ...] = (6, 12, 24),
    diff_offsets: tuple[int, ...] = (1, 6, 24),
    county_idx: int = 0,
    n_counties: int = 1,
) -> dict[str, float]:
    """
    Build a single feature row (dictionary) for one (county, timestamp) pair.

    Parameters
    ----------
    out_hist : 1-D array
        Historical outage counts up to (but not including) the target hour.
    tracked_hist : 1-D array
        Historical tracked-household counts (same length as *out_hist*).
    weather_hist : 2-D array, shape (T_hist, n_weather_features)
        Historical weather observations for this county.
    target_ts : pd.Timestamp
        The timestamp we are building features *for* (i.e. the forecast target).
    weather_name_to_idx : dict
        Mapping from weather variable name → column index in *weather_hist*.
    weather_features : list[str]
        Ordered list of all weather variable names (same order as columns of
        *weather_hist*).
    lags, rolling_windows, ... : feature-engineering knobs
        Typically loaded from the YAML config.
    county_idx, n_counties : int
        Used to create a normalised county-encoding feature.

    Returns
    -------
    dict[str, float]
        Feature name → value.  The caller stacks these into a DataFrame.
    """
    # Resolve defaults for optional subsets
    if weather_subset is None:
        weather_subset = DEFAULT_KEY_WEATHER
    if rolling_weather_subset is None:
        rolling_weather_subset = DEFAULT_ROLLING_WEATHER
    if diff_weather_subset is None:
        diff_weather_subset = DEFAULT_DIFF_WEATHER

    out_hist = np.asarray(out_hist, dtype=np.float32)
    tracked_hist = np.asarray(tracked_hist, dtype=np.float32)
    weather_hist = np.asarray(weather_hist, dtype=np.float32)

    # ── Most-recent values ──────────────────────────────────────────────
    latest_out = float(out_hist[-1]) if len(out_hist) else 0.0
    latest_tracked = max(float(tracked_hist[-1]) if len(tracked_hist) else 1.0, 1.0)
    latest_weather = (
        weather_hist[-1]
        if len(weather_hist)
        else np.zeros(len(weather_features), dtype=np.float32)
    )

    # ── Temporal encodings ──────────────────────────────────────────────
    hour = target_ts.hour
    hour_sin, hour_cos = _sin_cos(hour, 24)
    dow_sin, dow_cos = _sin_cos(target_ts.dayofweek, 7)
    m_sin, m_cos = _sin_cos(target_ts.month, 12)
    is_night = float(hour >= 21 or hour <= 5)

    feat: dict[str, float] = {
        "hour":         float(hour),
        "hour_sin":     hour_sin,
        "hour_cos":     hour_cos,
        "dow":          float(target_ts.dayofweek),
        "dow_sin":      dow_sin,
        "dow_cos":      dow_cos,
        "day_of_month": float(target_ts.day),
        "month":        float(target_ts.month),
        "month_sin":    m_sin,
        "month_cos":    m_cos,
        "is_weekend":   float(target_ts.dayofweek >= 5),
        "is_night":     is_night,
        # ── County / infrastructure ─────────────────────────────────────
        "tracked":      latest_tracked,
        "log_tracked":  float(np.log1p(latest_tracked)),
        "out_last":     latest_out,
        "outage_rate":  latest_out / latest_tracked,
        "county_idx":   float(county_idx) / max(n_counties - 1, 1),
    }

    # ── Outage lags ─────────────────────────────────────────────────────
    for lag in lags:
        lag_val = float(out_hist[-lag]) if len(out_hist) >= lag else latest_out
        feat[f"out_lag_{lag}"] = lag_val
        feat[f"outage_rate_lag_{lag}"] = lag_val / latest_tracked

    # ── Outage momentum (first differences at configurable offsets) ─────
    # Each diff_offset computes: outage_now - outage_{offset}_hours_ago.
    # Short offsets (1, 6) capture fast-moving storm trends; longer offsets
    # (48, 168) capture multi-day and week-over-week changes.
    for offset in diff_offsets:
        required_len = offset + 1  # need at least offset+1 data points
        feat[f"out_diff_{offset}"] = (
            float(out_hist[-1] - out_hist[-(offset + 1)])
            if len(out_hist) >= required_len
            else 0.0
        )

    # ── Outage acceleration (second difference) ─────────────────────────
    feat["out_accel"] = (
        float((out_hist[-1] - out_hist[-2]) - (out_hist[-2] - out_hist[-3]))
        if len(out_hist) >= 3
        else 0.0
    )

    # ── Zero-outage streak (hours since last nonzero outage) ────────────
    streak = 0
    for i in range(len(out_hist) - 1, -1, -1):
        if out_hist[i] > 0:
            break
        streak += 1
    feat["zero_streak"] = float(min(streak, 168))  # cap at 1 week

    # ── Outage percentile versus history ────────────────────────────────
    feat["out_pctile"] = (
        float(np.mean(out_hist <= latest_out)) if len(out_hist) >= 24 else 0.5
    )

    # ── Rolling outage statistics ───────────────────────────────────────
    all_roll_windows = sorted(set(rolling_windows) | {48})
    for window in all_roll_windows:
        arr = out_hist[-window:] if len(out_hist) >= window else out_hist
        if len(arr) == 0:
            feat[f"out_roll_mean_{window}"] = 0.0
            feat[f"out_roll_max_{window}"] = 0.0
            feat[f"out_roll_std_{window}"] = 0.0
        else:
            feat[f"out_roll_mean_{window}"] = float(np.mean(arr))
            feat[f"out_roll_max_{window}"] = float(np.max(arr))
            feat[f"out_roll_std_{window}"] = float(np.std(arr))

    # ── Exponential moving average (approximate) ────────────────────────
    for span in (6, 24):
        alpha_ema = 2.0 / (span + 1)
        if len(out_hist) >= span:
            chunk = out_hist[-span:]
            weights = np.array(
                [(1 - alpha_ema) ** i for i in range(span - 1, -1, -1)],
                dtype=np.float32,
            )
            weights /= weights.sum()
            feat[f"out_ema_{span}"] = float(np.dot(chunk, weights))
        else:
            feat[f"out_ema_{span}"] = latest_out

    # ── Raw weather features (all variables) ────────────────────────────
    for idx, name in enumerate(weather_features):
        feat[f"w_{name}"] = float(latest_weather[idx])

    # ── Lagged weather for key subset ───────────────────────────────────
    for name in weather_subset:
        if name not in weather_name_to_idx:
            continue
        idx = weather_name_to_idx[name]
        feat[f"w_{name}_last"] = float(latest_weather[idx])
        for lag in weather_lags:
            if len(weather_hist) >= lag:
                feat[f"w_{name}_lag_{lag}"] = float(weather_hist[-lag, idx])
            else:
                feat[f"w_{name}_lag_{lag}"] = float(latest_weather[idx])

    # ── Rolling weather statistics ──────────────────────────────────────
    for name in rolling_weather_subset:
        if name not in weather_name_to_idx:
            continue
        idx = weather_name_to_idx[name]
        for window in weather_rolling_windows:
            arr = (
                weather_hist[-window:, idx]
                if len(weather_hist) >= window
                else weather_hist[:, idx]
            )
            if len(arr) == 0:
                feat[f"w_{name}_roll_mean_{window}"] = 0.0
                feat[f"w_{name}_roll_max_{window}"] = 0.0
            else:
                feat[f"w_{name}_roll_mean_{window}"] = float(np.mean(arr))
                feat[f"w_{name}_roll_max_{window}"] = float(np.max(arr))

    # ── Weather first-difference features ───────────────────────────────
    for name in diff_weather_subset:
        if name not in weather_name_to_idx:
            continue
        idx = weather_name_to_idx[name]
        feat[f"w_{name}_diff_1"] = (
            float(weather_hist[-1, idx] - weather_hist[-2, idx])
            if len(weather_hist) >= 2
            else 0.0
        )
        feat[f"w_{name}_diff_6"] = (
            float(weather_hist[-1, idx] - weather_hist[-7, idx])
            if len(weather_hist) >= 7
            else 0.0
        )

    # ── Derived: wind speed and powers ──────────────────────────────────
    wind_speed = 0.0
    if "u10" in weather_name_to_idx and "v10" in weather_name_to_idx:
        u = float(latest_weather[weather_name_to_idx["u10"]])
        v = float(latest_weather[weather_name_to_idx["v10"]])
        wind_speed = math.sqrt(u * u + v * v)
    feat["wind_speed"] = wind_speed
    feat["wind_speed_sq"] = wind_speed * wind_speed
    feat["wind_speed_cu"] = wind_speed ** 3

    # ── Derived: peak ratio (recent max / historical max) ───────────────
    if len(out_hist) >= 24:
        recent_max = float(np.max(out_hist[-24:]))
        alltime_max = float(np.max(out_hist))
        feat["peak_ratio_24"] = recent_max / max(alltime_max, 1.0)
    else:
        feat["peak_ratio_24"] = 0.0
    feat["peak_ratio_48"] = (
        float(np.max(out_hist[-48:])) / max(float(np.max(out_hist)), 1.0)
        if len(out_hist) >= 48
        else feat["peak_ratio_24"]
    )

    # ── Derived: temperature extremes ───────────────────────────────────
    if "t2m" in weather_name_to_idx:
        t2m = float(latest_weather[weather_name_to_idx["t2m"]])
        feat["temp_extreme"] = (t2m - 288.0) ** 2
        feat["temp_cold"] = float(max(273.0 - t2m, 0.0))
        feat["temp_hot"] = float(max(t2m - 305.0, 0.0))
    else:
        feat["temp_extreme"] = 0.0
        feat["temp_cold"] = 0.0
        feat["temp_hot"] = 0.0

    # ── Derived: weather severity (z-score composite) ───────────────────
    sev, n_sev = 0.0, 0
    for sev_name in ("gust", "cape", "prate", "tp"):
        if sev_name in weather_name_to_idx:
            idx_s = weather_name_to_idx[sev_name]
            val = float(latest_weather[idx_s])
            if len(weather_hist) >= 24:
                col = weather_hist[-24:, idx_s]
                mu, sd = float(np.mean(col)), float(np.std(col)) + 1e-8
                sev += (val - mu) / sd
            else:
                sev += val
            n_sev += 1
    feat["weather_severity"] = sev / max(n_sev, 1)

    # ── Interaction terms: weather × outage ─────────────────────────────
    feat["wind_x_out"] = wind_speed * latest_out
    feat["wind_x_outrate"] = wind_speed * (latest_out / latest_tracked)
    if "gust" in weather_name_to_idx:
        gust = float(latest_weather[weather_name_to_idx["gust"]])
        feat["gust_x_out"] = gust * latest_out
    else:
        feat["gust_x_out"] = 0.0
    feat["severity_x_out"] = feat["weather_severity"] * latest_out
    feat["severity_x_momentum"] = feat["weather_severity"] * feat.get("out_diff_1", 0.0)
    gust_diff = feat.get("w_gust_diff_1", 0.0)
    feat["gust_diff_x_momentum"] = gust_diff * feat.get("out_diff_6", 0.0)
    feat["wind_sq_x_outrate"] = feat["wind_speed_sq"] * (latest_out / latest_tracked)

    # ── Interaction terms: time × weather ───────────────────────────────
    feat["night_wind"] = is_night * wind_speed
    feat["night_out"] = is_night * latest_out

    return feat


def dataset_to_tabular(
    ds,
    *,
    lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 12, 24, 48),
    rolling_windows: tuple[int, ...] = (6, 12, 24),
    weather_subset: list[str] | None = None,
    rolling_weather_subset: list[str] | None = None,
    diff_weather_subset: list[str] | None = None,
    weather_lags: tuple[int, ...] = (1, 3, 6, 12, 24),
    weather_rolling_windows: tuple[int, ...] = (6, 12, 24),
    diff_offsets: tuple[int, ...] = (1, 6, 24),
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert an xarray Dataset into a flat tabular feature matrix.

    This is the main entry point used by any model that needs tabular features
    (currently DKL; could also be used by XGBoost / LightGBM, etc.).

    Parameters
    ----------
    ds : xr.Dataset
        Must contain variables ``out``, ``tracked``, ``weather`` and
        dimensions ``timestamp``, ``location``, ``feature``.

    Returns
    -------
    X_df : pd.DataFrame
        Feature matrix — one row per (county, timestamp) pair.
    y : np.ndarray
        Target outage values, aligned with the rows of *X_df*.
    row_timestamps : np.ndarray of pd.Timestamp
        Timestamp for each row.
    row_locations : np.ndarray of str
        County identifier for each row.
    """
    timestamps = pd.to_datetime(ds.timestamp.values)
    locations = [str(loc) for loc in ds.location.values]
    weather_features = [str(x) for x in ds.feature.values]
    weather_name_to_idx = {name: i for i, name in enumerate(weather_features)}

    out = ds.out.transpose("timestamp", "location").values.astype(np.float32)
    tracked = ds.tracked.transpose("timestamp", "location").values.astype(np.float32)
    weather = ds.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)

    # Minimum history required before we can build a feature row
    min_history = max(max(lags), max(rolling_windows))

    rows: list[dict[str, float]] = []
    targets: list[float] = []
    row_ts_list: list[pd.Timestamp] = []
    row_loc_list: list[str] = []

    for c, loc in enumerate(locations):
        for t in range(min_history, len(timestamps)):
            feat = build_tabular_feature_row(
                out_hist=out[:t, c],
                tracked_hist=tracked[:t, c],
                weather_hist=weather[:t, c, :],
                target_ts=timestamps[t],
                weather_name_to_idx=weather_name_to_idx,
                weather_features=weather_features,
                lags=lags,
                rolling_windows=rolling_windows,
                weather_subset=weather_subset,
                rolling_weather_subset=rolling_weather_subset,
                diff_weather_subset=diff_weather_subset,
                weather_lags=weather_lags,
                weather_rolling_windows=weather_rolling_windows,
                diff_offsets=diff_offsets,
                county_idx=c,
                n_counties=len(locations),
            )
            rows.append(feat)
            targets.append(float(out[t, c]))
            row_ts_list.append(timestamps[t])
            row_loc_list.append(loc)

    X_df = pd.DataFrame(rows).fillna(0.0)
    y = np.asarray(targets, dtype=np.float32)
    row_timestamps = np.array(row_ts_list)
    row_locations = np.array(row_loc_list, dtype=object)

    return X_df, y, row_timestamps, row_locations


def outage_column_mask(feature_columns: list[str]) -> np.ndarray:
    """
    Boolean mask identifying columns that depend on past outage values.

    Used during noise-injection training: we add Gaussian noise only to
    outage-derived columns so that the model becomes robust to the
    compounding errors that arise in autoregressive prediction.
    """
    prefixes = (
        "out_lag_", "outage_rate_lag_", "out_roll_mean_", "out_roll_max_",
        "out_roll_std_", "out_last", "outage_rate",
    )
    return np.array(
        [any(c.startswith(p) for p in prefixes) for c in feature_columns],
        dtype=bool,
    )


# =========================================================================== #
#  2. Sequence feature helpers (Seq2Seq)                                       #
# =========================================================================== #

def z_normalize_fit(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute column-wise mean and standard deviation for z-normalisation.

    Parameters
    ----------
    arr : np.ndarray
        Input array; statistics are computed along axis 0.

    Returns
    -------
    mu, sd : np.ndarray
        Mean and standard deviation (with zeros replaced by 1.0 to avoid
        division-by-zero).
    """
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def z_normalize_apply(arr: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Apply pre-computed z-normalisation: (arr - mu) / sd."""
    return (arr - mu) / sd


def build_sliding_windows(
    X_loc: np.ndarray,
    y_loc: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window samples for one location.

    Parameters
    ----------
    X_loc : shape (T, D)
        Feature matrix for a single county.
    y_loc : shape (T,)
        Target outage series for a single county.
    seq_len : int
        Number of look-back time steps (encoder input length).
    horizon : int
        Number of forecast time steps (decoder output length).

    Returns
    -------
    X_windows : shape (N, seq_len, D)
    Y_windows : shape (N, horizon)
        where N = T - seq_len - horizon + 1.
    """
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return (
            np.empty((0, seq_len, X_loc.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    X_windows, Y_windows = [], []
    for i in range(N):
        X_windows.append(X_loc[i : i + seq_len])
        Y_windows.append(y_loc[i + seq_len : i + seq_len + horizon])

    return (
        np.asarray(X_windows, dtype=np.float32),
        np.asarray(Y_windows, dtype=np.float32),
    )


# =========================================================================== #
#  3. SARIMAX exogenous feature builders                                       #
# =========================================================================== #
#
# SARIMAX requires exogenous variables to be time-aligned arrays that are
# available for BOTH the historical (training) period AND the future
# (prediction) period.  This limits us to:
#   - Weather variables  (provided in the dataset for both train and test)
#   - Deterministic temporal encodings  (computable for any timestamp)
#   - Derived variables that combine the above (e.g. wind speed from u10/v10)
#
# Outage-derived features (lags, rolling stats) are NOT suitable here because
# they are unknown for the future horizon.  SARIMAX handles its own
# autoregressive lags internally via the (p, d, q) order.
# =========================================================================== #


def build_temporal_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build deterministic temporal features for a sequence of timestamps.

    These are always available (no dependency on observed data) and can be
    used as exogenous regressors during both training and forecasting.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex
        Ordered timestamps for which to build features.

    Returns
    -------
    pd.DataFrame
        Columns: hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos,
        is_weekend, is_night.  Index = *timestamps*.
    """
    hours = timestamps.hour
    dows = timestamps.dayofweek
    months = timestamps.month

    return pd.DataFrame({
        "hour_sin":   np.sin(2.0 * np.pi * hours / 24.0),
        "hour_cos":   np.cos(2.0 * np.pi * hours / 24.0),
        "dow_sin":    np.sin(2.0 * np.pi * dows / 7.0),
        "dow_cos":    np.cos(2.0 * np.pi * dows / 7.0),
        "month_sin":  np.sin(2.0 * np.pi * months / 12.0),
        "month_cos":  np.cos(2.0 * np.pi * months / 12.0),
        "is_weekend": (dows >= 5).astype(float),
        "is_night":   ((hours >= 21) | (hours <= 5)).astype(float),
    }, index=timestamps)


def build_sarimax_exog(
    ds,
    location: str,
    timestamps: pd.DatetimeIndex | None = None,
    weather_features: list[str] | None = None,
    include_temporal: bool = True,
    include_wind_speed: bool = True,
) -> np.ndarray | None:
    """
    Build an exogenous feature matrix for one county from an xarray Dataset.

    This is the main helper used by ``CountySARIMAX`` to extract the
    exogenous regressors that accompany the outage series during both
    training (from ``ds_train``) and forecasting (from ``ds_future``).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a ``weather`` variable with dims
        (timestamp, location, feature).
    location : str
        County identifier to select from the dataset.
    timestamps : pd.DatetimeIndex, optional
        If provided, the exogenous matrix is built for these timestamps
        only.  Defaults to all timestamps in *ds*.
    weather_features : list[str], optional
        Which weather variables to include.  ``None`` means no weather
        features (only temporal features will be used if enabled).
    include_temporal : bool
        Whether to append deterministic temporal encodings.
    include_wind_speed : bool
        Whether to derive wind speed from u10/v10 and append it.

    Returns
    -------
    np.ndarray or None
        Shape (T, n_features).  Returns ``None`` if no features were
        requested (both weather and temporal disabled).
    """
    if timestamps is None:
        timestamps = pd.to_datetime(ds.timestamp.values)

    frames: list[pd.DataFrame] = []

    # ── Weather features from the dataset ──────────────────────────────
    if weather_features:
        available = [str(f) for f in ds.feature.values]
        valid_features = [f for f in weather_features if f in available]

        if valid_features:
            # Extract weather array for this county: shape (T, n_features)
            weather_all = ds.weather.sel(location=location).transpose(
                "timestamp", "feature"
            )
            ds_timestamps = pd.to_datetime(ds.timestamp.values)

            weather_df = pd.DataFrame(
                {f: weather_all.sel(feature=f).values.astype(float) for f in valid_features},
                index=ds_timestamps,
            )
            # Align to the requested timestamps (forward-fill for safety)
            weather_df = weather_df.reindex(timestamps, method="ffill")
            frames.append(weather_df)

            # ── Derived: wind speed from u10 and v10 ──────────────────
            if include_wind_speed and "u10" in valid_features and "v10" in valid_features:
                ws = np.sqrt(weather_df["u10"] ** 2 + weather_df["v10"] ** 2)
                frames.append(pd.DataFrame({"wind_speed": ws}, index=timestamps))

    # ── Temporal features (deterministic — always available) ───────────
    if include_temporal:
        frames.append(build_temporal_features(timestamps))

    # ── Combine ────────────────────────────────────────────────────────
    if not frames:
        return None

    exog_df = pd.concat(frames, axis=1)
    return exog_df.values.astype(float)

