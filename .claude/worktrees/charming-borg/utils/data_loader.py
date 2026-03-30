"""
utils/data_loader.py
====================
Data-loading and validation-splitting utilities.

Key components
--------------
- ``load_competition_data``   : reads the train / test NetCDF files.
- ``WalkForwardFold``         : a single fold produced by walk-forward splitting.
- ``walk_forward_split``      : generates *K* expanding-window folds that mimic
                                k-fold cross-validation while respecting the
                                temporal ordering of the data.

Walk-forward validation
-----------------------
Standard k-fold CV is unsuitable for time-series because it allows future data
to leak into the training set.  Walk-forward validation fixes this:

    Fold 1:  [=====train=====][==val==]...............
    Fold 2:  [========train========][==val==]..........
    Fold 3:  [===========train===========][==val==]....
    ...
    Fold K:  [================train================][==val==]

Each fold uses an *expanding* training window; the validation window always has
a fixed size equal to the forecast horizon (48 hours).  This lets us tune
hyperparameters across multiple temporal slices — analogous to k-fold CV for
i.i.d. data — without look-ahead bias.

Reference: demo.ipynb cell 16 (original single temporal split).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


# =========================================================================== #
#  Data containers                                                             #
# =========================================================================== #

@dataclass
class CompetitionData:
    """Holds all data loaded from the competition NetCDF files."""

    ds_train: xr.Dataset                         # Full training dataset
    ds_test_48h: Optional[xr.Dataset]            # Demo test set (may be None)
    locations: list[str]                          # County identifiers
    weather_features: list[str]                   # Weather variable names
    train_timestamps: pd.DatetimeIndex            # Training timestamps
    test_48h_timestamps: Optional[pd.DatetimeIndex]  # Test timestamps


@dataclass
class WalkForwardFold:
    """
    A single fold from the walk-forward split.

    Attributes
    ----------
    fold_idx : int
        Zero-based fold index.
    ds_train_sub : xr.Dataset
        Training portion (all timestamps before the validation window).
    ds_val : xr.Dataset
        Validation portion (exactly ``horizon`` timestamps).
    val_timestamps : pd.DatetimeIndex
        The timestamps that define the validation window.
    val_truth : np.ndarray, shape (horizon, n_counties)
        Ground-truth outage matrix for the validation window.
    """

    fold_idx: int
    ds_train_sub: xr.Dataset
    ds_val: xr.Dataset
    val_timestamps: pd.DatetimeIndex
    val_truth: np.ndarray  # shape (horizon, n_counties)


# =========================================================================== #
#  Loading                                                                     #
# =========================================================================== #

def load_competition_data(
    data_dir: str | Path,
    train_file: str = "train.nc",
    test_48h_file: str = "test_48h_demo.nc",
) -> CompetitionData:
    """
    Load the competition datasets from *data_dir*.

    Parameters
    ----------
    data_dir : path-like
        Directory containing the NetCDF files.
    train_file : str
        Filename of the training set.
    test_48h_file : str
        Filename of the 48-hour demo test set (ignored if absent).
    """
    data_dir = Path(data_dir)

    ds_train = xr.open_dataset(data_dir / train_file)

    # The test file is optional — it may not be present during local dev.
    ds_test_48h: Optional[xr.Dataset] = None
    test_48h_timestamps: Optional[pd.DatetimeIndex] = None
    test_path = data_dir / test_48h_file
    if test_path.exists():
        ds_test_48h = xr.open_dataset(test_path)
        test_48h_timestamps = pd.to_datetime(ds_test_48h.timestamp.values)

    return CompetitionData(
        ds_train=ds_train,
        ds_test_48h=ds_test_48h,
        locations=[str(x) for x in ds_train.location.values],
        weather_features=[str(x) for x in ds_train.feature.values],
        train_timestamps=pd.to_datetime(ds_train.timestamp.values),
        test_48h_timestamps=test_48h_timestamps,
    )


# =========================================================================== #
#  Walk-forward validation                                                     #
# =========================================================================== #

def walk_forward_split(
    ds: xr.Dataset,
    n_folds: int = 3,
    horizon: int = 48,
    min_train_frac: float = 0.5,
) -> list[WalkForwardFold]:
    """
    Generate *n_folds* expanding-window walk-forward folds.

    Parameters
    ----------
    ds : xr.Dataset
        The full training dataset (with dimension ``timestamp``).
    n_folds : int
        Number of validation folds to create.
    horizon : int
        Number of timestamps in each validation window (must match the
        competition forecast horizon — typically 48 hours).
    min_train_frac : float
        Minimum fraction of the total timeline that the *first* fold must
        use for training.  Prevents degenerate folds where the model has
        very little history.

    Returns
    -------
    list[WalkForwardFold]
        One ``WalkForwardFold`` per fold, ordered chronologically.

    Raises
    ------
    ValueError
        If the dataset is too short or parameters are inconsistent.

    Example
    -------
    With 1000 timestamps, n_folds=3, horizon=48, min_train_frac=0.5:

        Fold 0 — train :   0 .. 499,  val : 500 .. 547
        Fold 1 — train :   0 .. 684,  val : 685 .. 732
        Fold 2 — train :   0 .. 903,  val : 904 .. 951
    """
    n_total = ds.sizes["timestamp"]

    # Minimum training size (in timestamps)
    min_train = max(int(n_total * min_train_frac), horizon + 1)

    # The last fold's validation window must end at or before the last timestamp
    # Available space for validation folds:  n_total - min_train
    available = n_total - min_train
    if available < horizon:
        raise ValueError(
            f"Dataset has {n_total} timestamps but min_train={min_train} and "
            f"horizon={horizon} leave no room for even one fold."
        )
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1.")

    # Space the validation-start indices evenly across the available range.
    # The last fold always ends at the dataset boundary.
    last_val_start = n_total - horizon
    first_val_start = min_train

    if n_folds == 1:
        val_starts = [last_val_start]
    else:
        val_starts = [
            int(round(first_val_start + i * (last_val_start - first_val_start) / (n_folds - 1)))
            for i in range(n_folds)
        ]

    folds: list[WalkForwardFold] = []
    for k, vs in enumerate(val_starts):
        ve = vs + horizon  # validation end (exclusive)
        ds_train_sub = ds.isel(timestamp=slice(0, vs))
        ds_val = ds.isel(timestamp=slice(vs, ve))

        val_ts = pd.to_datetime(ds_val.timestamp.values)
        val_truth = (
            ds_val.out.transpose("timestamp", "location")
            .values.astype(float)
        )

        folds.append(
            WalkForwardFold(
                fold_idx=k,
                ds_train_sub=ds_train_sub,
                ds_val=ds_val,
                val_timestamps=val_ts,
                val_truth=val_truth,
            )
        )

    return folds
