import math
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# --------------------------------------------------------------------------- #
# [ACTION 4] Expanded from 10 → 20 key weather features to match colleague's
#            notebook.  Adds cape_1, refc, hail, mstav, pwat, sh2, lftx, blh,
#            sdlwrf, pcdb — storm indicators, soil/boundary-layer, radiation.
# --------------------------------------------------------------------------- #
DEFAULT_KEY_WEATHER = [
    "gust", "cape", "cape_1", "tp", "prate", "refc", "hail",
    "u10", "v10", "t2m", "mstav", "pwat", "sh2", "lftx",
    "blh", "sdlwrf", "sp", "tcc", "r2", "pcdb",
]

# [ACTION 4] Subset used for rolling-weather aggregations
DEFAULT_ROLLING_WEATHER = [
    "gust", "cape", "tp", "prate", "t2m", "mstav",
]

# [ACTION 4] Subset used for weather first-difference features
DEFAULT_DIFF_WEATHER = [
    "gust", "cape", "t2m", "sp", "prate",
]


def _sin_cos(value: int | float, period: int | float) -> tuple[float, float]:
    angle = 2.0 * math.pi * float(value) / float(period)
    return math.sin(angle), math.cos(angle)


def _z_value(alpha: float) -> float:
    if abs(alpha - 0.05) < 1e-12:
        return 1.959963984540054
    if abs(alpha - 0.10) < 1e-12:
        return 1.6448536269514722
    if abs(alpha - 0.01) < 1e-12:
        return 2.5758293035489004
    return 1.959963984540054


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 128, 64),
        embed_dim: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _ApproximateDKLGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        feature_extractor: nn.Module,
        # [ACTION 1] Default raised to 768 to match colleague
        num_inducing: int = 768,
        embed_dim: int = 20,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        inducing_points = torch.randn(num_inducing, embed_dim, device=device)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=embed_dim)
        )

    def forward(self, embedded_x):
        mean = self.mean_module(embedded_x)
        covar = self.covar_module(embedded_x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def __call__(self, x, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, **kwargs)


@dataclass
class DKLCalibration:
    q_by_county: np.ndarray
    alpha: float = 0.05


class DKLForecaster:
    def __init__(
        self,
        # [ACTION 6] Added lags 4, 5 to match colleague's 9-lag set
        lags: Iterable[int] = (1, 2, 3, 4, 5, 6, 12, 24, 48),
        rolling_windows: Iterable[int] = (6, 12, 24),
        weather_subset: Optional[Iterable[str]] = None,
        rolling_weather_subset: Optional[Iterable[str]] = None,
        diff_weather_subset: Optional[Iterable[str]] = None,
        hidden_dims: Iterable[int] = (256, 128, 64),
        embed_dim: int = 20,
        # [ACTION 1] Increased from 512 → 768 to match colleague
        num_inducing: int = 768,
        dropout: float = 0.1,
        # [ACTION 5] Increased from 25 → 40 to match colleague
        epochs: int = 40,
        # [ACTION 8] Increased from 512 → 1024 to match colleague
        batch_size: int = 1024,
        lr: float = 5e-3,
        # [ACTION 2] Removed training-row cap — use all data by default
        max_train_rows: Optional[int] = None,
        random_state: int = 42,
        device: Optional[str] = None,
        clip_nonnegative: bool = True,
        num_workers: int = 0,
        predict_batch_size: int = 4096,
        # [ACTION 3] New: gradient clipping max norm
        grad_clip_norm: float = 1.0,
        # [ACTION 6] New: weather lag steps — added 3 and 12
        weather_lags: Iterable[int] = (1, 3, 6, 12, 24),
        # [ACTION 6] New: weather rolling windows — added 12
        weather_rolling_windows: Iterable[int] = (6, 12, 24),
    ):
        self.lags = tuple(sorted(set(int(x) for x in lags)))
        self.rolling_windows = tuple(sorted(set(int(x) for x in rolling_windows)))
        self.weather_subset = (
            list(weather_subset) if weather_subset is not None else list(DEFAULT_KEY_WEATHER)
        )
        self.rolling_weather_subset = (
            list(rolling_weather_subset)
            if rolling_weather_subset is not None
            else list(DEFAULT_ROLLING_WEATHER)
        )
        self.diff_weather_subset = (
            list(diff_weather_subset)
            if diff_weather_subset is not None
            else list(DEFAULT_DIFF_WEATHER)
        )
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.embed_dim = int(embed_dim)
        self.num_inducing = int(num_inducing)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.max_train_rows = max_train_rows
        self.random_state = int(random_state)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_nonnegative = bool(clip_nonnegative)
        self.num_workers = int(num_workers)
        self.predict_batch_size = int(predict_batch_size)
        self.grad_clip_norm = float(grad_clip_norm)
        self.weather_lags = tuple(sorted(set(int(x) for x in weather_lags)))
        self.weather_rolling_windows = tuple(sorted(set(int(x) for x in weather_rolling_windows)))

        self.pin_memory = self.device.type == "cuda"

        self.x_scaler = StandardScaler()
        self.y_mean_ = 0.0
        self.y_std_ = 1.0

        self.feature_columns_: list[str] = []
        self.locations_: list[str] = []
        self.weather_features_: list[str] = []
        self.input_dim_: Optional[int] = None

        self.model = None
        self.likelihood = None

        self.calibration_: Optional[DKLCalibration] = None
        self.train_history_: list[float] = []

    @property
    def min_history(self) -> int:
        return max(max(self.lags), max(self.rolling_windows))

    def _build_feature_row(
        self,
        out_hist,
        tracked_hist,
        weather_hist,
        target_ts,
        weather_name_to_idx,
    ):
        out_hist = np.asarray(out_hist, dtype=np.float32)
        tracked_hist = np.asarray(tracked_hist, dtype=np.float32)
        weather_hist = np.asarray(weather_hist, dtype=np.float32)

        latest_out = float(out_hist[-1]) if len(out_hist) else 0.0
        latest_tracked = float(tracked_hist[-1]) if len(tracked_hist) else 1.0
        latest_tracked = max(latest_tracked, 1.0)

        latest_weather = (
            weather_hist[-1]
            if len(weather_hist)
            else np.zeros(len(self.weather_features_), dtype=np.float32)
        )

        hour_sin, hour_cos = _sin_cos(target_ts.hour, 24)
        dow_sin, dow_cos = _sin_cos(target_ts.dayofweek, 7)

        feat = {
            "hour": float(target_ts.hour),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow": float(target_ts.dayofweek),
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            # [ACTION 6] Added day_of_month to match colleague
            "day_of_month": float(target_ts.day),
            "month": float(target_ts.month),
            "is_weekend": float(target_ts.dayofweek >= 5),
            "tracked": latest_tracked,
            "log_tracked": float(np.log1p(latest_tracked)),
            "out_last": latest_out,
            "outage_rate": latest_out / latest_tracked,
        }

        # Outage lags — now includes 4 and 5
        for lag in self.lags:
            lag_val = float(out_hist[-lag]) if len(out_hist) >= lag else latest_out
            feat[f"out_lag_{lag}"] = lag_val
            feat[f"outage_rate_lag_{lag}"] = lag_val / latest_tracked

        # Rolling outage stats
        for window in self.rolling_windows:
            arr = out_hist[-window:] if len(out_hist) >= window else out_hist
            if len(arr) == 0:
                feat[f"out_roll_mean_{window}"] = 0.0
                feat[f"out_roll_max_{window}"] = 0.0
                feat[f"out_roll_std_{window}"] = 0.0
            else:
                feat[f"out_roll_mean_{window}"] = float(np.mean(arr))
                feat[f"out_roll_max_{window}"] = float(np.max(arr))
                feat[f"out_roll_std_{window}"] = float(np.std(arr))

        # Raw weather features (all)
        for idx, name in enumerate(self.weather_features_):
            feat[f"w_{name}"] = float(latest_weather[idx])

        # ------------------------------------------------------------------
        # Lagged weather for key subset
        # [ACTION 6] Now uses self.weather_lags = (1, 3, 6, 12, 24)
        # ------------------------------------------------------------------
        for name in self.weather_subset:
            if name not in weather_name_to_idx:
                continue

            idx = weather_name_to_idx[name]
            feat[f"w_{name}_last"] = float(latest_weather[idx])

            for lag in self.weather_lags:
                if len(weather_hist) >= lag:
                    feat[f"w_{name}_lag_{lag}"] = float(weather_hist[-lag, idx])
                else:
                    feat[f"w_{name}_lag_{lag}"] = float(latest_weather[idx])

        # ------------------------------------------------------------------
        # Rolling weather stats for rolling subset
        # [ACTION 6] Now uses self.weather_rolling_windows = (6, 12, 24)
        # ------------------------------------------------------------------
        for name in self.rolling_weather_subset:
            if name not in weather_name_to_idx:
                continue

            idx = weather_name_to_idx[name]
            for window in self.weather_rolling_windows:
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

        # ------------------------------------------------------------------
        # [ACTION 4] NEW: Weather first-difference features (diff_1, diff_6).
        #            Captures rate-of-change — critical for predicting
        #            outage onset during rapidly worsening conditions.
        # ------------------------------------------------------------------
        for name in self.diff_weather_subset:
            if name not in weather_name_to_idx:
                continue

            idx = weather_name_to_idx[name]
            # 1-hour difference
            if len(weather_hist) >= 2:
                feat[f"w_{name}_diff_1"] = float(
                    weather_hist[-1, idx] - weather_hist[-2, idx]
                )
            else:
                feat[f"w_{name}_diff_1"] = 0.0

            # 6-hour difference
            if len(weather_hist) >= 7:
                feat[f"w_{name}_diff_6"] = float(
                    weather_hist[-1, idx] - weather_hist[-7, idx]
                )
            else:
                feat[f"w_{name}_diff_6"] = 0.0

        # Wind speed (derived)
        if "u10" in weather_name_to_idx and "v10" in weather_name_to_idx:
            u = float(latest_weather[weather_name_to_idx["u10"]])
            v = float(latest_weather[weather_name_to_idx["v10"]])
            feat["wind_speed"] = math.sqrt(u * u + v * v)

        return feat

    def _dataset_to_training_table(self, ds):
        timestamps = pd.to_datetime(ds.timestamp.values)
        locations = [str(loc) for loc in ds.location.values]

        self.weather_features_ = [str(x) for x in ds.feature.values]
        weather_name_to_idx = {name: i for i, name in enumerate(self.weather_features_)}

        out = ds.out.transpose("timestamp", "location").values.astype(np.float32)
        tracked = ds.tracked.transpose("timestamp", "location").values.astype(np.float32)
        weather = ds.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)

        rows = []
        targets = []
        row_locs = []

        for c, loc in enumerate(locations):
            for t in range(self.min_history, len(timestamps)):
                feat = self._build_feature_row(
                    out_hist=out[:t, c],
                    tracked_hist=tracked[:t, c],
                    weather_hist=weather[:t, c, :],
                    target_ts=timestamps[t],
                    weather_name_to_idx=weather_name_to_idx,
                )
                rows.append(feat)
                targets.append(float(out[t, c]))
                row_locs.append(loc)

        X_df = pd.DataFrame(rows).fillna(0.0)
        y = np.asarray(targets, dtype=np.float32)
        locs = np.asarray(row_locs, dtype=object)
        return X_df, y, locs

    def _init_model(self, input_dim: int):
        feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            embed_dim=self.embed_dim,
            dropout=self.dropout,
        ).to(self.device)

        model = _ApproximateDKLGP(
            feature_extractor=feature_extractor,
            num_inducing=self.num_inducing,
            embed_dim=self.embed_dim,
            device=self.device,
        ).to(self.device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        return model, likelihood

    def fit(self, ds):
        self.locations_ = [str(loc) for loc in ds.location.values]

        X_df, y, _ = self._dataset_to_training_table(ds)
        self.feature_columns_ = X_df.columns.tolist()

        X = X_df.to_numpy(dtype=np.float32)

        # [ACTION 2] max_train_rows defaults to None — use all data
        if self.max_train_rows is not None and len(X) > self.max_train_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_rows, replace=False)
            X = X[idx]
            y = y[idx]

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_scaled = self.x_scaler.fit_transform(X).astype(np.float32)

        y_log = np.log1p(np.clip(y, 0.0, None)).astype(np.float32)
        self.y_mean_ = float(y_log.mean())
        self.y_std_ = float(y_log.std() + 1e-8)
        y_scaled = ((y_log - self.y_mean_) / self.y_std_).astype(np.float32)

        self.input_dim_ = X_scaled.shape[1]
        self.model, self.likelihood = self._init_model(self.input_dim_)

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_scaled, dtype=torch.float32),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.feature_extractor.parameters(), "lr": self.lr},
                {"params": self.model.variational_parameters(), "lr": self.lr * 2.0},
                {"params": self.model.mean_module.parameters(), "lr": self.lr},
                {"params": self.model.covar_module.parameters(), "lr": self.lr},
                {"params": self.likelihood.parameters(), "lr": self.lr},
            ]
        )

        # [ACTION 3] CosineAnnealingLR scheduler to match colleague
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood,
            self.model,
            num_data=len(dataset),
        )

        self.train_history_ = []
        print(f"[DKL] training on device: {self.device}")
        print(f"[DKL] features: {self.input_dim_}, samples: {len(dataset)}")

        for epoch in range(self.epochs):
            start = time.time()
            epoch_loss = 0.0
            n_batch = 0

            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=self.pin_memory)
                yb = yb.to(self.device, non_blocking=self.pin_memory)

                optimizer.zero_grad(set_to_none=True)
                output = self.model(xb)
                loss = -mll(output, yb)
                loss.backward()

                # [ACTION 3] Gradient clipping to prevent GP explosions
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

            # [ACTION 3] Step the scheduler
            scheduler.step()

            avg_loss = epoch_loss / n_batch
            self.train_history_.append(avg_loss)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[DKL] epoch={epoch + 1}/{self.epochs} "
                f"loss={avg_loss:.6f} lr={current_lr:.6f} "
                f"time={time.time() - start:.2f}s"
            )

        return self

    @torch.no_grad()
    def _predict_distribution(self, X_raw: np.ndarray):
        X_scaled = self.x_scaler.transform(
            np.asarray(X_raw, dtype=np.float32)
        ).astype(np.float32)

        self.model.eval()
        self.likelihood.eval()

        means_scaled = []
        stds_scaled = []

        with gpytorch.settings.fast_pred_var():
            for start in range(0, len(X_scaled), self.predict_batch_size):
                stop = start + self.predict_batch_size
                xb = torch.tensor(
                    X_scaled[start:stop], dtype=torch.float32
                ).to(self.device)
                pred_dist = self.likelihood(self.model(xb))

                means_scaled.append(pred_dist.mean.detach().cpu().numpy())
                stds_scaled.append(pred_dist.stddev.detach().cpu().numpy())

        mean_scaled = np.concatenate(means_scaled, axis=0)
        std_scaled = np.concatenate(stds_scaled, axis=0)

        mean_log = mean_scaled * self.y_std_ + self.y_mean_
        std_log = np.maximum(std_scaled * abs(self.y_std_), 1e-6)
        return mean_log, std_log

    def calibrate_intervals(
        self,
        ds,
        calibration_size: int = 48,
        alpha: float = 0.05,
    ):
        n = ds.sizes["timestamp"]
        if n <= calibration_size + self.min_history:
            raise ValueError("Not enough timestamps for calibration.")

        ds_fit = ds.isel(timestamp=slice(0, n - calibration_size))
        ds_cal = ds.isel(timestamp=slice(n - calibration_size, n))

        clone = self.__class__(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            weather_subset=self.weather_subset,
            rolling_weather_subset=self.rolling_weather_subset,
            diff_weather_subset=self.diff_weather_subset,
            hidden_dims=self.hidden_dims,
            embed_dim=self.embed_dim,
            num_inducing=self.num_inducing,
            dropout=self.dropout,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            max_train_rows=self.max_train_rows,
            random_state=self.random_state,
            device=str(self.device),
            clip_nonnegative=self.clip_nonnegative,
            num_workers=self.num_workers,
            predict_batch_size=self.predict_batch_size,
            grad_clip_norm=self.grad_clip_norm,
            weather_lags=self.weather_lags,
            weather_rolling_windows=self.weather_rolling_windows,
        )
        clone.fit(ds_fit)

        cal_timestamps = pd.to_datetime(
            ds_cal.timestamp.values[:calibration_size]
        )
        cal_truth = (
            ds_cal.out.transpose("timestamp", "location")
            .isel(timestamp=slice(0, calibration_size))
            .values.astype(np.float32)
        )

        pred_df = clone.predict(ds_fit, cal_timestamps, return_intervals=False)

        pred_matrix = (
            pred_df.assign(
                timestamp=pd.to_datetime(pred_df["timestamp"]),
                location=pred_df["location"].astype(str),
            )
            .pivot(index="timestamp", columns="location", values="pred")
            .reindex(
                index=cal_timestamps,
                columns=[str(x) for x in self.locations_],
            )
            .to_numpy(dtype=np.float32)
        )

        abs_resid = np.abs(cal_truth - pred_matrix)

        # [ACTION 7] Finite-sample correction for conformal quantile
        n_cal = abs_resid.shape[0]
        conformal_level = min(
            np.ceil((1.0 - alpha) * (n_cal + 1)) / n_cal,
            1.0,
        )
        q = np.quantile(abs_resid, conformal_level, axis=0)

        self.calibration_ = DKLCalibration(q_by_county=q, alpha=alpha)
        return self.calibration_

    def predict(
        self,
        ds_context,
        timestamps,
        return_intervals: bool = False,
        alpha: float = 0.05,
        interval_method: str = "posterior",
    ):
        if self.model is None or self.likelihood is None:
            raise ValueError("Model has not been fitted yet.")

        timestamps = pd.to_datetime(timestamps)
        locations = [str(loc) for loc in ds_context.location.values]
        weather_name_to_idx = {
            name: i for i, name in enumerate(self.weather_features_)
        }

        out = ds_context.out.transpose("timestamp", "location").values.astype(
            np.float32
        )
        tracked = ds_context.tracked.transpose(
            "timestamp", "location"
        ).values.astype(np.float32)
        weather = ds_context.weather.transpose(
            "timestamp", "location", "feature"
        ).values.astype(np.float32)

        out_histories = [
            list(out[:, c].astype(float)) for c in range(out.shape[1])
        ]
        tracked_histories = [
            list(tracked[:, c].astype(float)) for c in range(tracked.shape[1])
        ]
        weather_histories = [
            weather[:, c, :].astype(np.float32).copy()
            for c in range(weather.shape[1])
        ]

        all_rows = []
        z = _z_value(alpha)

        for ts in timestamps:
            step_features = []
            for c in range(len(locations)):
                feat = self._build_feature_row(
                    out_hist=out_histories[c],
                    tracked_hist=tracked_histories[c],
                    weather_hist=weather_histories[c],
                    target_ts=ts,
                    weather_name_to_idx=weather_name_to_idx,
                )
                step_features.append(
                    [feat.get(col, 0.0) for col in self.feature_columns_]
                )

            X_step = np.asarray(step_features, dtype=np.float32)
            mean_log, std_log = self._predict_distribution(X_step)

            pred = np.expm1(mean_log)
            if self.clip_nonnegative:
                pred = np.clip(pred, 0.0, None)

            if return_intervals:
                if (
                    interval_method == "conformal"
                    and self.calibration_ is not None
                ):
                    q = self.calibration_.q_by_county
                    lower = np.clip(pred - q, 0.0, None)
                    upper = pred + q
                else:
                    lower = np.expm1(mean_log - z * std_log)
                    upper = np.expm1(mean_log + z * std_log)
                    lower = np.clip(lower, 0.0, None)
                    upper = np.maximum(upper, lower)
            else:
                lower = upper = None

            for c, loc in enumerate(locations):
                row = {
                    "timestamp": ts,
                    "location": loc,
                    "pred": float(pred[c]),
                }
                if return_intervals:
                    row["lower"] = float(lower[c])
                    row["upper"] = float(upper[c])
                all_rows.append(row)

            for c in range(len(locations)):
                out_histories[c].append(float(pred[c]))
                tracked_histories[c].append(float(tracked_histories[c][-1]))
                last_weather = weather_histories[c][-1:, :]
                weather_histories[c] = np.vstack(
                    [weather_histories[c], last_weather]
                )

        return pd.DataFrame(all_rows)