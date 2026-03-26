import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def z_normalize_fit(arr):
    mu = np.nanmean(arr, axis=0)
    sd = np.nanstd(arr, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def z_normalize_apply(arr, mu, sd):
    return (arr - mu) / sd


def build_sliding_windows(X_loc, y_loc, seq_len, horizon):
    N = len(y_loc) - seq_len - horizon + 1
    if N <= 0:
        return (
            np.empty((0, seq_len, X_loc.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    X_windows, Y_windows = [], []
    for i in range(N):
        X_windows.append(X_loc[i:i + seq_len])
        Y_windows.append(y_loc[i + seq_len:i + seq_len + horizon])

    return np.asarray(X_windows, dtype=np.float32), np.asarray(Y_windows, dtype=np.float32)


class Seq2SeqWindowDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class SimpleSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, horizon=48):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h_last = h[-1]
        return self.head(h_last)


class Seq2SeqForecaster:
    def __init__(
        self,
        seq_len=24,
        horizon=48,
        hidden_dim=64,
        num_layers=1,
        batch_size=64,
        epochs=5,
        lr=1e-3,
        device=None,
        clip_nonnegative=True,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_nonnegative = clip_nonnegative

        self.model = None
        self.scalers = None
        self.input_dim = None
        self.locations_ = None

    def _prepare_training_arrays(self, ds):
        y = ds.out.transpose("timestamp", "location").values.astype(float)
        w = ds.weather.transpose("timestamp", "location", "feature").values.astype(float)

        T, L, F = w.shape

        y_mu, y_sd = z_normalize_fit(y.reshape(-1, 1))
        w_mu, w_sd = z_normalize_fit(w.reshape(-1, F))

        y_scaled = z_normalize_apply(y.reshape(-1, 1), y_mu, y_sd).reshape(T, L)
        w_scaled = z_normalize_apply(w.reshape(-1, F), w_mu, w_sd).reshape(T, L, F)

        input_dim = 1 + F
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

        scalers = {
            "y_mu": y_mu,
            "y_sd": y_sd,
            "w_mu": w_mu,
            "w_sd": w_sd,
        }

        return X, Y, input_dim, scalers

    def _prepare_context_inputs(self, ds_context):
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

    def fit(self, ds):
        self.locations_ = [str(loc) for loc in ds.location.values]

        X, Y, input_dim, scalers = self._prepare_training_arrays(ds)
        if len(X) == 0:
            raise ValueError("No training windows were created. Check seq_len and horizon.")

        self.scalers = scalers
        self.input_dim = input_dim

        dataset = Seq2SeqWindowDataset(X, Y)
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
            start = time.time()

            for xb, yb in dataloader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            epoch_loss /= len(dataset)
            print(f"[Seq2Seq] epoch={epoch+1}/{self.epochs} loss={epoch_loss:.6f} time={time.time()-start:.2f}s")

        return self

    @torch.no_grad()
    def predict(self, ds_context, timestamps):
        if self.model is None or self.scalers is None:
            raise ValueError("Model has not been fitted yet.")

        timestamps = pd.to_datetime(timestamps)
        X_context, ordered_locations = self._prepare_context_inputs(ds_context)

        X_tensor = torch.tensor(X_context, dtype=torch.float32).to(self.device)
        pred_scaled = self.model(X_tensor).cpu().numpy()  # (L, horizon)

        y_mu = self.scalers["y_mu"].flatten()[0]
        y_sd = self.scalers["y_sd"].flatten()[0]
        pred = pred_scaled * y_sd + y_mu

        if self.clip_nonnegative:
            pred = np.clip(pred, 0, None)

        rows = []
        for i, loc in enumerate(ordered_locations):
            rows.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "location": str(loc),
                        "pred": pred[i],
                    }
                )
            )

        return pd.concat(rows, ignore_index=True)