from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


@dataclass
class CompetitionData:
    ds_train: xr.Dataset
    ds_test_48h: Optional[xr.Dataset]
    locations: list[str]
    weather_features: list[str]
    train_timestamps: pd.DatetimeIndex
    test_48h_timestamps: Optional[pd.DatetimeIndex]


@dataclass
class ValidationSplit:
    ds_train_sub: xr.Dataset
    ds_val: xr.Dataset
    val_timestamps_48h: pd.DatetimeIndex
    val_truth_48h: np.ndarray


def load_competition_data(
    data_dir: str | Path,
    train_file: str = "train.nc",
    test_48h_file: str = "test_48h_demo.nc",
) -> CompetitionData:
    data_dir = Path(data_dir)

    ds_train = xr.open_dataset(data_dir / train_file)

    ds_test_48h = None
    test_48h_timestamps = None
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


def temporal_split(ds: xr.Dataset, val_fraction: float = 0.2) -> ValidationSplit:
    n = ds.sizes["timestamp"]
    split_idx = int(n * (1 - val_fraction))

    ds_train_sub = ds.isel(timestamp=slice(0, split_idx))
    ds_val = ds.isel(timestamp=slice(split_idx, None))

    val_timestamps_48h = pd.to_datetime(ds_val.timestamp.values[:48])
    val_truth_48h = (
        ds_val.out.transpose("timestamp", "location")
        .isel(timestamp=slice(0, 48))
        .values.astype(float)
    )

    return ValidationSplit(
        ds_train_sub=ds_train_sub,
        ds_val=ds_val,
        val_timestamps_48h=val_timestamps_48h,
        val_truth_48h=val_truth_48h,
    )