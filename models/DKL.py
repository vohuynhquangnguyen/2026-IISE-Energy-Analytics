"""
models/dkl.py
=============
Deep Kernel Learning (DKL) forecaster for power-outage prediction.

Architecture
------------
- **Feature extractor**: a multi-layer MLP with BatchNorm + Dropout that
  compresses the high-dimensional tabular feature vector into a low-dimensional
  embedding.
- **Gaussian Process head**: a sparse variational GP (VariationalStrategy +
  CholeskyVariationalDistribution) operates on the embedding to produce both a
  mean prediction and a calibrated uncertainty estimate.

The tabular features are built by ``utils.feature_engineering.dataset_to_tabular``
(shared with all models) and include outage lags, rolling stats, weather
features, interactions, and temporal encodings.

Configuration is loaded from ``configs/dkl.yaml`` via ``from_config()``.

Reference: colleague's DKL notebook, demo.ipynb baseline logic.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils.config import load_config
from utils.feature_engineering import (
    DEFAULT_DIFF_WEATHER,
    DEFAULT_KEY_WEATHER,
    DEFAULT_ROLLING_WEATHER,
    build_tabular_feature_row,
    dataset_to_tabular,
    outage_column_mask,
)


# =========================================================================== #
#  Small helpers                                                               #
# =========================================================================== #

def _z_value(alpha: float) -> float:
    """Return the standard-normal z-value for a two-sided (1 - alpha) CI."""
    lookup = {0.05: 1.959963984540054, 0.10: 1.6448536269514722, 0.01: 2.5758293035489004}
    return lookup.get(round(alpha, 4), 1.959963984540054)


# =========================================================================== #
#  Neural-network feature extractor                                            #
# =========================================================================== #

class FeatureExtractor(nn.Module):
    """
    MLP that maps a tabular feature vector to a low-dimensional embedding
    consumed by the GP kernel.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (256, 128, 64),
        embed_dim: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =========================================================================== #
#  Sparse Variational GP on top of the feature extractor                       #
# =========================================================================== #

class _ApproximateDKLGP(gpytorch.models.ApproximateGP):
    """
    Variational GP that first passes inputs through a learned feature
    extractor, then applies an ARD-RBF kernel on the embedding space.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        num_inducing: int = 768,
        embed_dim: int = 20,
        device: Optional[torch.device] = None,
    ):
        device = device or torch.device("cpu")
        inducing_points = torch.randn(num_inducing, embed_dim, device=device)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
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
            gpytorch.kernels.RBFKernel(ard_num_dims=embed_dim),
        )

    def forward(self, embedded_x):
        mean = self.mean_module(embedded_x)
        covar = self.covar_module(embedded_x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def __call__(self, x, **kwargs):
        x = self.feature_extractor(x)
        return super().__call__(x, **kwargs)


# =========================================================================== #
#  Conformal calibration data container                                        #
# =========================================================================== #

@dataclass
class DKLCalibration:
    """County-specific conformal interval half-widths."""
    q_by_county: np.ndarray
    alpha: float = 0.05


# =========================================================================== #
#  Per-Horizon Normalized Conformal with ACI (improvements 1+2+3)              #
# =========================================================================== #

class PerHorizonConformal:
    """
    Combines three high-impact improvements for DKL prediction intervals:

    1. **Normalized conformal** — uses GP posterior std as the score normalizer,
       producing variable-width intervals (tight where confident, wide where
       uncertain): score_i = |y_i - mu_i| / sigma_i.

    2. **Log-space calibration** — calibrates in log1p space and back-transforms
       via expm1(), yielding naturally asymmetric intervals that match the
       heavy-tailed outage distribution.

    3. **Per-horizon calibration with ACI** — maintains separate residual pools
       and adaptive alpha per forecast step-ahead, so intervals naturally widen
       with horizon.  Adaptive Conformal Inference (Gibbs & Candes 2021) adjusts
       the effective significance level based on recent miscoverage.

    All three compose: normalized scores in log-space, stored per-horizon,
    with ACI alpha adaptation.
    """

    def __init__(
        self,
        horizon: int = 48,
        alpha: float = 0.05,
        window: int = 300,
        aci_lr: float = 0.005,
    ):
        self.horizon = horizon
        self.base_alpha = alpha
        self.aci_lr = aci_lr
        # Separate residual pools per horizon step (signed normalized residuals)
        self.residuals: dict[int, deque] = {
            h: deque(maxlen=window) for h in range(horizon)
        }
        # Adaptive alpha per horizon step (ACI)
        self.alpha_h: dict[int, float] = {h: alpha for h in range(horizon)}

    def update(
        self,
        log_preds: np.ndarray,
        log_stds: np.ndarray,
        log_actuals: np.ndarray,
    ) -> None:
        """
        Feed in one complete forecast cycle (up to horizon steps).

        All inputs are in log1p space.  ``log_stds`` comes from the GP
        posterior and is used to normalize residuals.

        Parameters
        ----------
        log_preds : array, shape (H,) or (H, C)
            Predicted means in log1p space.
        log_stds : array, shape (H,) or (H, C)
            Predicted GP std in log1p space.
        log_actuals : array, shape (H,) or (H, C)
            Actual values in log1p space.
        """
        n_steps = min(len(log_preds), self.horizon)
        for h in range(n_steps):
            p = np.atleast_1d(log_preds[h])
            s = np.atleast_1d(log_stds[h])
            a = np.atleast_1d(log_actuals[h])
            # Signed normalized residuals (in log-space)
            signed_norm = (a - p) / np.maximum(s, 1e-6)
            for val in signed_norm.ravel():
                self.residuals[h].append(float(val))

    def get_intervals(
        self,
        log_preds: np.ndarray,
        log_stds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return calibrated prediction intervals in *original* (count) space.

        Uses per-horizon asymmetric quantiles of signed normalized residuals,
        computed in log-space, then back-transformed via expm1().

        Parameters
        ----------
        log_preds : array, shape (H,) or (H, C)
            Predicted means in log1p space.
        log_stds : array, shape (H,) or (H, C)
            GP posterior std in log1p space.

        Returns
        -------
        lower, upper : arrays in original count space
        """
        H = min(len(log_preds), self.horizon)
        lower = np.zeros_like(log_preds[:H], dtype=float)
        upper = np.zeros_like(log_preds[:H], dtype=float)

        for h in range(H):
            res = np.array(self.residuals[h])
            n = len(res)

            # If insufficient data for this horizon, pool from nearby
            if n < 20:
                nearby: list[float] = []
                for hh in range(max(0, h - 3), min(self.horizon, h + 4)):
                    nearby.extend(self.residuals[hh])
                res = np.array(nearby)
                n = len(res)

            if n < 5:
                # Extreme fallback: use 2× GP std
                log_lo = log_preds[h] - 2.0 * log_stds[h]
                log_hi = log_preds[h] + 2.0 * log_stds[h]
            else:
                a = self.alpha_h[h]
                # Asymmetric quantiles of signed normalized residuals
                q_lo = np.quantile(res, a / 2)
                q_hi = np.quantile(res, min(np.ceil((n + 1) * (1 - a / 2)) / n, 1.0))
                # Intervals in log-space (naturally asymmetric after expm1)
                log_lo = log_preds[h] + q_lo * log_stds[h]
                log_hi = log_preds[h] + q_hi * log_stds[h]

            # Back-transform to original space
            lower[h] = np.maximum(np.expm1(log_lo), 0.0)
            upper[h] = np.expm1(log_hi)

        return lower, upper

    def update_aci(
        self,
        log_preds: np.ndarray,
        log_stds: np.ndarray,
        log_actuals: np.ndarray,
    ) -> None:
        """
        ACI adaptive alpha update after observing actuals.

        α_{h,t+1} = α_{h,t} + γ · (err_t - α_base)
        where err_t = 1 if observation fell outside the interval, 0 otherwise.
        """
        n_steps = min(len(log_preds), self.horizon)
        for h in range(n_steps):
            res = np.array(self.residuals[h])
            if len(res) < 10:
                continue

            a = self.alpha_h[h]
            q_lo = np.quantile(res, a / 2)
            q_hi = np.quantile(res, min(1 - a / 2, 1.0))

            p = np.atleast_1d(log_preds[h])
            s = np.atleast_1d(log_stds[h])
            act = np.atleast_1d(log_actuals[h])
            norm = (act - p) / np.maximum(s, 1e-6)

            # Check coverage for each county
            covered = np.mean((norm >= q_lo) & (norm <= q_hi))
            err = 1.0 - covered

            # ACI update: if under-covered, alpha decreases → wider intervals
            self.alpha_h[h] = float(np.clip(
                a + self.aci_lr * (err - self.base_alpha),
                0.005,
                0.20,
            ))


# =========================================================================== #
#  DKL Forecaster                                                              #
# =========================================================================== #

class DKLForecaster:
    """
    End-to-end DKL forecaster: feature engineering → MLP → GP → forecast.

    All feature-engineering knobs (lags, rolling windows, weather subsets) are
    stored here so that the same settings are applied at both train and predict
    time.  The actual feature computation is delegated to
    ``utils.feature_engineering``.

    Parameters
    ----------
    lags, rolling_windows, weather_subset, ... : feature settings
        Passed through to ``build_tabular_feature_row`` /
        ``dataset_to_tabular`` at both fit and predict time.
    hidden_dims, embed_dim, num_inducing, dropout : model architecture
    epochs, batch_size, lr, grad_clip_norm : training loop settings
    lag_noise_frac, lag_noise_scale : noise-injection settings for
        robustifying the model to autoregressive error accumulation.
    """

    def __init__(
        self,
        # --- Feature engineering ---
        lags: Iterable[int] = (1, 2, 3, 4, 5, 6, 12, 24, 48),
        rolling_windows: Iterable[int] = (6, 12, 24),
        weather_subset: Optional[Iterable[str]] = None,
        rolling_weather_subset: Optional[Iterable[str]] = None,
        diff_weather_subset: Optional[Iterable[str]] = None,
        weather_lags: Iterable[int] = (1, 3, 6, 12, 24),
        weather_rolling_windows: Iterable[int] = (6, 12, 24),
        diff_offsets: Iterable[int] = (1, 6, 24),
        # --- Model architecture ---
        hidden_dims: Iterable[int] = (256, 128, 64),
        embed_dim: int = 20,
        num_inducing: int = 768,
        dropout: float = 0.1,
        # --- Training ---
        epochs: int = 40,
        batch_size: int = 1024,
        lr: float = 5e-3,
        grad_clip_norm: float = 1.0,
        max_train_rows: Optional[int] = None,
        random_state: int = 42,
        device: Optional[str] = None,
        clip_nonnegative: bool = True,
        num_workers: int = 0,
        predict_batch_size: int = 4096,
        lag_noise_frac: float = 0.3,
        lag_noise_scale: float = 0.15,
        early_stopping_patience: int = 10,
    ):
        # Feature engineering settings (forwarded to shared module)
        self.lags = tuple(sorted(set(int(x) for x in lags)))
        self.rolling_windows = tuple(sorted(set(int(x) for x in rolling_windows)))
        self.weather_subset = (
            list(weather_subset) if weather_subset is not None
            else list(DEFAULT_KEY_WEATHER)
        )
        self.rolling_weather_subset = (
            list(rolling_weather_subset) if rolling_weather_subset is not None
            else list(DEFAULT_ROLLING_WEATHER)
        )
        self.diff_weather_subset = (
            list(diff_weather_subset) if diff_weather_subset is not None
            else list(DEFAULT_DIFF_WEATHER)
        )
        self.weather_lags = tuple(sorted(set(int(x) for x in weather_lags)))
        self.weather_rolling_windows = tuple(sorted(set(int(x) for x in weather_rolling_windows)))
        self.diff_offsets = tuple(sorted(set(int(x) for x in diff_offsets)))

        # Model architecture
        self.hidden_dims = tuple(int(x) for x in hidden_dims)
        self.embed_dim = int(embed_dim)
        self.num_inducing = int(num_inducing)
        self.dropout = float(dropout)

        # Training
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.grad_clip_norm = float(grad_clip_norm)
        self.max_train_rows = max_train_rows
        self.random_state = int(random_state)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.num_workers = int(num_workers)
        self.predict_batch_size = int(predict_batch_size)
        self.lag_noise_frac = float(lag_noise_frac)
        self.lag_noise_scale = float(lag_noise_scale)
        self.early_stopping_patience = int(early_stopping_patience)

        # Device selection
        if device is None or device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.pin_memory = self.device.type == "cuda"

        # Fitted state
        self.x_scaler = StandardScaler()
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.feature_columns_: list[str] = []
        self.locations_: list[str] = []
        self.weather_features_: list[str] = []
        self.input_dim_: Optional[int] = None
        self.model = None
        self.likelihood = None
        self.calibration_: Optional[DKLCalibration] = None
        self.train_history_: list[float] = []

    # ------------------------------------------------------------------ #
    #  Factory: build from YAML config                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config_path: str | Path = "dkl.yaml") -> "DKLForecaster":
        """Instantiate a DKLForecaster from a YAML configuration file."""
        cfg = load_config(config_path)
        feat_cfg = cfg.get("features", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})

        # Resolve device
        raw_device = train_cfg.get("device", "auto")

        return cls(
            # Features
            lags=feat_cfg.get("lags", [1, 2, 3, 4, 5, 6, 12, 24, 48]),
            rolling_windows=feat_cfg.get("rolling_windows", [6, 12, 24]),
            weather_subset=feat_cfg.get("weather_subset", None),
            rolling_weather_subset=feat_cfg.get("rolling_weather_subset", None),
            diff_weather_subset=feat_cfg.get("diff_weather_subset", None),
            weather_lags=feat_cfg.get("weather_lags", [1, 3, 6, 12, 24]),
            weather_rolling_windows=feat_cfg.get("weather_rolling_windows", [6, 12, 24]),
            diff_offsets=feat_cfg.get("diff_offsets", [1, 6, 24]),
            # Architecture
            hidden_dims=model_cfg.get("hidden_dims", [256, 128, 64]),
            embed_dim=model_cfg.get("embed_dim", 20),
            num_inducing=model_cfg.get("num_inducing", 768),
            dropout=model_cfg.get("dropout", 0.1),
            clip_nonnegative=model_cfg.get("clip_nonnegative", True),
            # Training
            epochs=train_cfg.get("epochs", 40),
            batch_size=train_cfg.get("batch_size", 1024),
            lr=train_cfg.get("lr", 5e-3),
            grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
            max_train_rows=train_cfg.get("max_train_rows", None),
            random_state=train_cfg.get("random_state", 42),
            device=raw_device,
            num_workers=train_cfg.get("num_workers", 0),
            predict_batch_size=train_cfg.get("predict_batch_size", 4096),
            lag_noise_frac=train_cfg.get("lag_noise_frac", 0.3),
            lag_noise_scale=train_cfg.get("lag_noise_scale", 0.15),
            early_stopping_patience=train_cfg.get("early_stopping_patience", 10),
        )

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def min_history(self) -> int:
        """Minimum number of historical time steps needed to build a feature row."""
        return max(max(self.lags), max(self.rolling_windows))

    @property
    def _feature_kwargs(self) -> dict:
        """Feature-engineering keyword arguments forwarded to shared helpers."""
        return dict(
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            weather_subset=self.weather_subset,
            rolling_weather_subset=self.rolling_weather_subset,
            diff_weather_subset=self.diff_weather_subset,
            weather_lags=self.weather_lags,
            weather_rolling_windows=self.weather_rolling_windows,
            diff_offsets=self.diff_offsets,
        )

    # ------------------------------------------------------------------ #
    #  Feature table construction (delegates to shared module)             #
    # ------------------------------------------------------------------ #

    def build_feature_table(self, ds):
        """
        Build a tabular feature matrix from the full dataset using
        ground-truth outage values (oracle features).

        Returns (X_df, y, row_timestamps, row_locations).
        """
        self.weather_features_ = [str(x) for x in ds.feature.values]
        X_df, y, row_ts, row_locs = dataset_to_tabular(ds, **self._feature_kwargs)
        return X_df, y, row_ts, row_locs

    def _dataset_to_training_table(self, ds):
        """Build features and save metadata needed for prediction."""
        self.weather_features_ = [str(x) for x in ds.feature.values]
        X_df, y, _, row_locs = dataset_to_tabular(ds, **self._feature_kwargs)
        return X_df, y, row_locs

    # ------------------------------------------------------------------ #
    #  Model initialisation                                                #
    # ------------------------------------------------------------------ #

    def _init_model(self, input_dim: int):
        """Create a fresh FeatureExtractor + GP model pair."""
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

    # ------------------------------------------------------------------ #
    #  Training loop (standard — from xarray Dataset)                      #
    # ------------------------------------------------------------------ #

    def fit(self, ds) -> "DKLForecaster":
        """
        Train the DKL model on the given xarray Dataset.

        Steps:
        1. Build the tabular feature matrix via the shared feature module.
        2. Optionally subsample rows.
        3. Scale features and log-transform targets.
        4. Run the variational ELBO training loop with cosine annealing.
        """
        self.locations_ = [str(loc) for loc in ds.location.values]

        # 1) Feature engineering (shared)
        X_df, y, _ = self._dataset_to_training_table(ds)
        self.feature_columns_ = X_df.columns.tolist()
        X = X_df.to_numpy(dtype=np.float32)

        # 2) Optional row subsampling
        if self.max_train_rows is not None and len(X) > self.max_train_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_rows, replace=False)
            X, y = X[idx], y[idx]

        # 3) Clean, scale, and log-transform
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.x_scaler.fit_transform(X).astype(np.float32)

        y_log = np.log1p(np.clip(y, 0.0, None)).astype(np.float32)
        self.y_mean_ = float(y_log.mean())
        self.y_std_ = float(y_log.std() + 1e-8)
        y_scaled = ((y_log - self.y_mean_) / self.y_std_).astype(np.float32)

        # 4) Initialise model and training infrastructure
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

        optimizer = torch.optim.Adam([
            {"params": self.model.feature_extractor.parameters(), "lr": self.lr},
            {"params": self.model.variational_parameters(), "lr": self.lr * 2.0},
            {"params": self.model.mean_module.parameters(), "lr": self.lr},
            {"params": self.model.covar_module.parameters(), "lr": self.lr},
            {"params": self.likelihood.parameters(), "lr": self.lr},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01,
        )
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=len(dataset),
        )

        self.train_history_ = []
        print(f"[DKL] device={self.device}, features={self.input_dim_}, samples={len(dataset)}")

        for epoch in range(self.epochs):
            t0 = time.time()
            epoch_loss, n_batch = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=self.pin_memory)
                yb = yb.to(self.device, non_blocking=self.pin_memory)

                optimizer.zero_grad(set_to_none=True)
                loss = -mll(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item()
                n_batch += 1

            scheduler.step()
            avg_loss = epoch_loss / n_batch
            self.train_history_.append(avg_loss)

            current_lr = scheduler.get_last_lr()[0]
            print(
                f"[DKL] epoch={epoch + 1}/{self.epochs}  "
                f"loss={avg_loss:.6f}  lr={current_lr:.6f}  "
                f"time={time.time() - t0:.2f}s"
            )

        return self

    # ------------------------------------------------------------------ #
    #  Training from pre-built arrays (oracle / noise-injection mode)      #
    # ------------------------------------------------------------------ #

    def fit_from_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val_X: Optional[np.ndarray] = None,
        val_y: Optional[np.ndarray] = None,
    ) -> "DKLForecaster":
        """
        Train from pre-built feature arrays with optional noise injection
        on outage-lag columns and early stopping on a validation set.

        This path is used by the oracle evaluation and walk-forward folds
        where features are already computed.
        """
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Optional row subsampling
        if self.max_train_rows is not None and len(X) > self.max_train_rows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_train_rows, replace=False)
            X, y = X[idx], y[idx]

        # Noise injection on outage-derived columns (robustness to AR errors)
        if self.lag_noise_frac > 0 and len(self.feature_columns_) > 0:
            cmask = outage_column_mask(self.feature_columns_)
            nc = cmask.sum()
            if nc > 0:
                rng = np.random.default_rng(self.random_state + 1)
                nn_rows = int(len(X) * self.lag_noise_frac)
                rows_idx = rng.choice(len(X), size=nn_rows, replace=False)
                stds = np.std(X[:, cmask], axis=0) + 1e-8
                noise = rng.normal(
                    0.0, self.lag_noise_scale * stds, size=(nn_rows, nc)
                ).astype(np.float32)
                X[np.ix_(rows_idx, np.where(cmask)[0])] += noise
                print(f"[DKL] noise: {nc} cols, {nn_rows:,}/{len(X):,} samples")

        # Scale and log-transform
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
            dataset, batch_size=self.batch_size, shuffle=True,
            pin_memory=self.pin_memory, num_workers=self.num_workers,
            persistent_workers=(self.num_workers > 0),
        )

        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([
            {"params": self.model.feature_extractor.parameters(), "lr": self.lr},
            {"params": self.model.variational_parameters(), "lr": self.lr * 2},
            {"params": self.model.mean_module.parameters(), "lr": self.lr},
            {"params": self.model.covar_module.parameters(), "lr": self.lr},
            {"params": self.likelihood.parameters(), "lr": self.lr},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01,
        )
        mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=len(dataset),
        )

        self.train_history_ = []
        best_val_rmse, best_state, best_lik_state = float("inf"), None, None
        patience_counter = 0

        print(f"[DKL] device={self.device}, features={self.input_dim_}, samples={len(dataset)}")
        for epoch in range(self.epochs):
            self.model.train()
            self.likelihood.train()
            t0 = time.time()
            eloss, nb = 0.0, 0

            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=self.pin_memory)
                yb = yb.to(self.device, non_blocking=self.pin_memory)
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()
                eloss += loss.item()
                nb += 1

            scheduler.step()
            avg = eloss / nb
            self.train_history_.append(avg)

            # Early stopping on validation set
            val_msg = ""
            if val_X is not None and val_y is not None:
                vr = self.predict_oracle(val_X)
                vrmse = np.sqrt(np.mean((val_y - vr["mean"]) ** 2))
                val_msg = f"  val_rmse={vrmse:.2f}"
                if vrmse < best_val_rmse:
                    best_val_rmse = vrmse
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    best_lik_state = {k: v.cpu().clone() for k, v in self.likelihood.state_dict().items()}
                    patience_counter = 0
                    val_msg += " *"
                else:
                    patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  epoch {epoch + 1}/{self.epochs}  loss={avg:.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.6f}  "
                    f"{time.time() - t0:.1f}s{val_msg}"
                )

            if patience_counter >= self.early_stopping_patience and val_X is not None:
                print(f"  Early stopping at epoch {epoch + 1} (best val_rmse={best_val_rmse:.2f})")
                break

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.likelihood.load_state_dict(best_lik_state)
            print(f"  Restored best model (val_rmse={best_val_rmse:.2f})")

        return self

    # ------------------------------------------------------------------ #
    #  Inference helpers                                                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _predict_distribution(self, X_raw: np.ndarray):
        """
        Forward pass through the scaler → MLP → GP pipeline.

        Returns (mean_log, std_log) — both in log1p-space so the caller
        can expm1() to get outage counts.
        """
        X_scaled = self.x_scaler.transform(
            np.asarray(X_raw, dtype=np.float32)
        ).astype(np.float32)

        self.model.eval()
        self.likelihood.eval()

        means_scaled, stds_scaled = [], []

        with gpytorch.settings.fast_pred_var():
            for start in range(0, len(X_scaled), self.predict_batch_size):
                stop = start + self.predict_batch_size
                xb = torch.tensor(X_scaled[start:stop], dtype=torch.float32).to(self.device)
                pred_dist = self.likelihood(self.model(xb))
                means_scaled.append(pred_dist.mean.detach().cpu().numpy())
                stds_scaled.append(pred_dist.stddev.detach().cpu().numpy())

        mean_scaled = np.concatenate(means_scaled, axis=0)
        std_scaled = np.concatenate(stds_scaled, axis=0)

        mean_log = mean_scaled * self.y_std_ + self.y_mean_
        std_log = np.maximum(std_scaled * abs(self.y_std_), 1e-6)
        return mean_log, std_log

    def predict_oracle(self, X: np.ndarray, ci_multiplier: float = 2.0) -> dict:
        """
        Single forward-pass prediction from pre-built features.

        Returns dict with keys "mean", "lower", "upper" (all in original
        outage-count space).
        """
        if self.model is None:
            raise ValueError("Model not fitted.")
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        ml, sl = self._predict_distribution(X)
        return {
            "mean": np.clip(np.expm1(ml), 0, None),
            "lower": np.clip(np.expm1(ml - ci_multiplier * sl), 0, None),
            "upper": np.clip(np.expm1(ml + ci_multiplier * sl), 0, None),
        }

    # ------------------------------------------------------------------ #
    #  Conformal interval calibration                                      #
    # ------------------------------------------------------------------ #

    def calibrate_intervals(
        self,
        ds,
        calibration_size: int = 48,
        alpha: float = 0.05,
    ) -> DKLCalibration:
        """
        Estimate county-specific conformal interval widths by training a
        clone model on the earlier portion of *ds* and evaluating residuals
        on the trailing calibration block.
        """
        n = ds.sizes["timestamp"]
        if n <= calibration_size + self.min_history:
            raise ValueError("Not enough timestamps for calibration.")

        ds_fit = ds.isel(timestamp=slice(0, n - calibration_size))
        ds_cal = ds.isel(timestamp=slice(n - calibration_size, n))

        # Train a fresh clone on the fit portion
        clone = self.__class__(
            lags=self.lags, rolling_windows=self.rolling_windows,
            weather_subset=self.weather_subset,
            rolling_weather_subset=self.rolling_weather_subset,
            diff_weather_subset=self.diff_weather_subset,
            hidden_dims=self.hidden_dims, embed_dim=self.embed_dim,
            num_inducing=self.num_inducing, dropout=self.dropout,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            max_train_rows=self.max_train_rows, random_state=self.random_state,
            device=str(self.device), clip_nonnegative=self.clip_nonnegative,
            num_workers=self.num_workers,
            predict_batch_size=self.predict_batch_size,
            grad_clip_norm=self.grad_clip_norm,
            weather_lags=self.weather_lags,
            weather_rolling_windows=self.weather_rolling_windows,
        )
        clone.fit(ds_fit)

        # Predict on the calibration block
        cal_timestamps = pd.to_datetime(ds_cal.timestamp.values[:calibration_size])
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
            .reindex(index=cal_timestamps, columns=[str(x) for x in self.locations_])
            .to_numpy(dtype=np.float32)
        )

        abs_resid = np.abs(cal_truth - pred_matrix)

        # Finite-sample correction for conformal quantile
        n_cal = abs_resid.shape[0]
        conformal_level = min(np.ceil((1.0 - alpha) * (n_cal + 1)) / n_cal, 1.0)
        q = np.quantile(abs_resid, conformal_level, axis=0)

        self.calibration_ = DKLCalibration(q_by_county=q, alpha=alpha)
        return self.calibration_

    # ------------------------------------------------------------------ #
    #  Autoregressive prediction (from xarray context)                     #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        ds_context,
        timestamps,
        return_intervals: bool = False,
        alpha: float = 0.05,
        interval_method: str = "posterior",
        return_log_space: bool = False,
    ) -> pd.DataFrame:
        """
        Autoregressive multi-step forecast.

        At each forecast step the model builds features from the available
        history (including its own prior predictions for the outage lags),
        makes a one-step prediction, and appends it to the history buffer.

        Parameters
        ----------
        ds_context : xr.Dataset
            Historical data up to the forecast origin.
        timestamps : array-like of datetime
            Target timestamps to forecast.
        return_intervals : bool
            Whether to include lower/upper columns.
        alpha : float
            Significance level for intervals.
        interval_method : str
            "conformal" (uses calibration) or "posterior" (GP variance).
        return_log_space : bool
            If True, also include ``log_pred`` and ``log_std`` columns
            (in log1p space) for use by PerHorizonConformal.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: timestamp, location, pred,
            [lower, upper], [log_pred, log_std].
        """
        if self.model is None or self.likelihood is None:
            raise ValueError("Model has not been fitted yet.")

        timestamps = pd.to_datetime(timestamps)
        locations = [str(loc) for loc in ds_context.location.values]
        weather_name_to_idx = {name: i for i, name in enumerate(self.weather_features_)}

        out = ds_context.out.transpose("timestamp", "location").values.astype(np.float32)
        tracked = ds_context.tracked.transpose("timestamp", "location").values.astype(np.float32)
        weather = ds_context.weather.transpose("timestamp", "location", "feature").values.astype(np.float32)

        # Initialise per-county history buffers
        out_histories = [list(out[:, c].astype(float)) for c in range(out.shape[1])]
        tracked_histories = [list(tracked[:, c].astype(float)) for c in range(tracked.shape[1])]
        weather_histories = [weather[:, c, :].astype(np.float32).copy() for c in range(weather.shape[1])]

        all_rows: list[dict] = []
        z = _z_value(alpha)

        for ts in timestamps:
            # Build one feature row per county
            step_features = []
            for c in range(len(locations)):
                feat = build_tabular_feature_row(
                    out_hist=out_histories[c],
                    tracked_hist=tracked_histories[c],
                    weather_hist=weather_histories[c],
                    target_ts=ts,
                    weather_name_to_idx=weather_name_to_idx,
                    weather_features=self.weather_features_,
                    county_idx=c,
                    n_counties=len(locations),
                    **self._feature_kwargs,
                )
                step_features.append(
                    [feat.get(col, 0.0) for col in self.feature_columns_]
                )

            X_step = np.asarray(step_features, dtype=np.float32)
            mean_log, std_log = self._predict_distribution(X_step)

            pred = np.expm1(mean_log)
            if self.clip_nonnegative:
                pred = np.clip(pred, 0.0, None)

            # Compute intervals
            if return_intervals:
                if interval_method == "conformal" and self.calibration_ is not None:
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
                row: dict = {"timestamp": ts, "location": loc, "pred": float(pred[c])}
                if return_intervals:
                    row["lower"] = float(lower[c])
                    row["upper"] = float(upper[c])
                if return_log_space:
                    row["log_pred"] = float(mean_log[c])
                    row["log_std"] = float(std_log[c])
                all_rows.append(row)

            # Update history buffers with the predictions (autoregressive)
            for c in range(len(locations)):
                out_histories[c].append(float(pred[c]))
                tracked_histories[c].append(float(tracked_histories[c][-1]))
                last_weather = weather_histories[c][-1:, :]
                weather_histories[c] = np.vstack([weather_histories[c], last_weather])

        return pd.DataFrame(all_rows)
