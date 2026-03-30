"""
models/seq2seq.py
=================
LSTM-based Sequence-to-Sequence forecaster for power-outage prediction.

Architecture
------------
- **Encoder**: multi-layer LSTM processes a look-back window of
  [scaled_outage, scaled_weather] features per county.
- **Decoder**: a single linear head maps the encoder's final hidden state
  to the full forecast horizon in one shot.

Feature engineering is delegated to ``utils.feature_engineering`` (z-norm
helpers and sliding-window construction), so every model shares the same
preprocessing pipeline.

Configuration is loaded from ``configs/seq2seq.yaml`` via the ``from_config``
class method.

Reference: demo.ipynb — Cells 23-26
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.config import load_config
from utils.feature_engineering import (
    build_sliding_windows,
    z_normalize_apply,
    z_normalize_fit,
)


# =========================================================================== #
#  PyTorch dataset wrapper                                                     #
# =========================================================================== #

class _Seq2SeqWindowDataset(Dataset):
    """Simple wrapper so sliding-window arrays can be consumed by a DataLoader."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


# =========================================================================== #
#  LSTM Seq2Seq network                                                        #
# =========================================================================== #

class SimpleSeq2Seq(nn.Module):
    """
    Minimal sequence-to-sequence model.

    The LSTM encoder encodes the look-back window; the last hidden state is
    projected through a linear head to produce the full forecast horizon.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        horizon: int = 48,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        _, (h, _) = self.lstm(x)          # h: (num_layers, batch, hidden_dim)
        h_last = h[-1]                     # (batch, hidden_dim)
        return self.head(h_last)           # (batch, horizon)


# =========================================================================== #
#  Seq2Seq forecaster (train / predict API)                                    #
# =========================================================================== #

class Seq2SeqForecaster:
    """
    High-level forecaster wrapping ``SimpleSeq2Seq``.

    Parameters
    ----------
    seq_len : int
        Number of look-back time steps (encoder input length).
    horizon : int
        Number of forecast time steps.
    hidden_dim, num_layers : int
        LSTM architecture settings.
    batch_size, epochs, lr : training settings.
    grad_clip_norm : float
        Maximum gradient norm (stabilises LSTM training).
    clip_nonnegative : bool
        Clip predictions to [0, ∞).
    device : str or None
        ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        seq_len: int = 24,
        horizon: int = 48,
        hidden_dim: int = 64,
        num_layers: int = 1,
        batch_size: int = 64,
        epochs: int = 100,
        lr: float = 1e-3,
        grad_clip_norm: float = 5.0,
        clip_nonnegative: bool = True,
        device: str | None = None,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.grad_clip_norm = grad_clip_norm
        self.clip_nonnegative = clip_nonnegative
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Filled after fit()
        self.model: Optional[SimpleSeq2Seq] = None
        self.scalers: Optional[dict] = None
        self.input_dim: Optional[int] = None
        self.locations_: Optional[list[str]] = None

    # ------------------------------------------------------------------ #
    #  Factory: build from YAML config                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config_path: str | Path = "seq2seq.yaml") -> "Seq2SeqForecaster":
        """
        Instantiate a Seq2SeqForecaster from a YAML configuration file.

        Parameters
        ----------
        config_path : str or Path
            Path (or filename) of the YAML config.
        """
        cfg = load_config(config_path)
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("training", {})

        return cls(
            seq_len=model_cfg.get("seq_len", 24),
            horizon=model_cfg.get("horizon", 48),
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 1),
            clip_nonnegative=model_cfg.get("clip_nonnegative", True),
            batch_size=train_cfg.get("batch_size", 64),
            epochs=train_cfg.get("epochs", 5),
            lr=train_cfg.get("lr", 1e-3),
            grad_clip_norm=train_cfg.get("grad_clip_norm", 5.0),
        )

    # ------------------------------------------------------------------ #
    #  Internal data preparation                                           #
    # ------------------------------------------------------------------ #

    def _prepare_training_arrays(self, ds):
        """
        Extract outage + weather arrays, z-normalise, and build sliding windows
        across all counties.

        Uses shared helpers from ``utils.feature_engineering``.
        """
        y = ds.out.transpose("timestamp", "location").values.astype(float)
        w = ds.weather.transpose("timestamp", "location", "feature").values.astype(float)
        T, L, F = w.shape

        # Fit global scalers on the training data
        y_mu, y_sd = z_normalize_fit(y.reshape(-1, 1))
        w_mu, w_sd = z_normalize_fit(w.reshape(-1, F))

        y_scaled = z_normalize_apply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, L)
        w_scaled = z_normalize_apply(w.reshape(-1, F), w_mu, w_sd).reshape(T, L, F)

        input_dim = 1 + F  # [outage, weather_features]
        X_list, Y_list = [], []

        for li in range(L):
            y_loc = y_scaled[:, li]
            w_loc = w_scaled[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1, 1), w_loc], axis=1)

            X_win, Y_win = build_sliding_windows(
                X_loc=X_loc,
                y_loc=y_loc,
                seq_len=self.seq_len,
                horizon=self.horizon,
            )
            if len(X_win) > 0:
                X_list.append(X_win)
                Y_list.append(Y_win)

        X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, self.seq_len, input_dim), dtype=np.float32)
        Y = np.concatenate(Y_list, axis=0) if Y_list else np.empty((0, self.horizon), dtype=np.float32)

        scalers = {"y_mu": y_mu, "y_sd": y_sd, "w_mu": w_mu, "w_sd": w_sd}
        return X, Y, input_dim, scalers

    def _prepare_context_inputs(self, ds_context):
        """
        Build the most-recent look-back window for each county (used at
        prediction time).
        """
        y = ds_context.out.transpose("timestamp", "location").values.astype(float)
        w = ds_context.weather.transpose("timestamp", "location", "feature").values.astype(float)
        T, L, F = w.shape

        if T < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} timestamps, got {T}.")

        y_scaled = z_normalize_apply(
            y.reshape(-1, 1), self.scalers["y_mu"], self.scalers["y_sd"]
        ).reshape(T, L)
        w_scaled = z_normalize_apply(
            w.reshape(-1, F), self.scalers["w_mu"], self.scalers["w_sd"]
        ).reshape(T, L, F)

        X_context = []
        for li in range(L):
            y_loc = y_scaled[:, li]
            w_loc = w_scaled[:, li, :]
            X_loc = np.concatenate([y_loc.reshape(-1, 1), w_loc], axis=1)
            X_context.append(X_loc[-self.seq_len:])

        X_context = np.asarray(X_context, dtype=np.float32)
        locations = [str(loc) for loc in ds_context.location.values]
        return X_context, locations

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, ds) -> "Seq2SeqForecaster":
        """
        Train the Seq2Seq model on the given xarray Dataset.

        Parameters
        ----------
        ds : xr.Dataset
            Training data with variables ``out`` and ``weather``.
        """
        self.locations_ = [str(loc) for loc in ds.location.values]

        X, Y, input_dim, scalers = self._prepare_training_arrays(ds)
        if len(X) == 0:
            raise ValueError("No training windows created. Check seq_len and horizon.")

        self.scalers = scalers
        self.input_dim = input_dim

        dataset = _Seq2SeqWindowDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = SimpleSeq2Seq(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            horizon=self.horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            print(
                f"[Seq2Seq] epoch={epoch + 1}/{self.epochs}  "
                f"loss={epoch_loss:.6f}  time={time.time() - t0:.2f}s"
            )

        return self

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(self, ds_context, timestamps) -> pd.DataFrame:
        """
        Forecast the next *horizon* timestamps using *ds_context* as history.

        Parameters
        ----------
        ds_context : xr.Dataset
            Historical data (the model uses the last ``seq_len`` timestamps).
        timestamps : array-like of datetime
            Target timestamps to forecast.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: timestamp, location, pred.
        """
        if self.model is None or self.scalers is None:
            raise ValueError("Model has not been fitted yet.")

        timestamps = pd.to_datetime(timestamps)
        X_context, ordered_locations = self._prepare_context_inputs(ds_context)

        X_tensor = torch.tensor(X_context, dtype=torch.float32).to(self.device)
        pred_scaled = self.model(X_tensor).cpu().numpy()  # (L, horizon)

        # Inverse z-transform on the outage predictions
        y_mu = self.scalers["y_mu"].flatten()[0]
        y_sd = self.scalers["y_sd"].flatten()[0]
        pred = pred_scaled * y_sd + y_mu

        if self.clip_nonnegative:
            pred = np.clip(pred, 0, None)

        rows: list[pd.DataFrame] = []
        for i, loc in enumerate(ordered_locations):
            rows.append(
                pd.DataFrame({
                    "timestamp": timestamps,
                    "location": str(loc),
                    "pred": pred[i],
                })
            )

        return pd.concat(rows, ignore_index=True)
