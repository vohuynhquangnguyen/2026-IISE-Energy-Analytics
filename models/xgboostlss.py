"""
models/xgboostlss.py
====================
XGBoostLSS forecaster with Zero-Inflated Negative Binomial (ZINB) distribution
for power-outage prediction.

Architecture
------------
- **Distribution**: ZINB — handles the heavy zero-inflation typical of outage
  data while modelling the over-dispersed positive counts.
- **Training strategy**: Direct multi-horizon — the forecast horizon is an
  explicit input feature so that a single model handles all 48 lead times.
- **Hyperparameter tuning**: Optuna-based Bayesian search with 3-fold CV.
- **Prediction intervals**: Monte-Carlo samples from the fitted distributional
  parameters, summarised as mean (point) and configurable quantiles (PI).

Feature engineering is centralised in ``utils.feature_engineering`` (Section 4)
so that training and inference use identical feature pipelines.

Configuration is loaded from ``configs/xgboostlss.yaml`` via ``from_config()``.

Reference: colleague's XGBoostLSS + SARIMAX ensemble notebook (v5).
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from utils.config import load_config
from utils.feature_engineering import (
    xgblss_build_forecast,
    xgblss_build_train,
    xgblss_feature_names,
)


class XGBoostLSSForecaster:
    """
    XGBoostLSS forecaster with ZINB distribution.

    Follows the same interface as the other models in this repo:
    ``from_config()`` → ``fit(ds)`` → ``predict(ds_context, timestamps)``.
    """

    def __init__(
        self,
        *,
        # Feature engineering
        lookback_lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        key_weather_features: list[str] | None = None,
        weather_rolling_windows: list[int] | None = None,
        weather_trend_top_n: int = 10,
        # Training data
        max_samples: int = 600_000,
        train_horizons: list[int] | None = None,
        max_horizon: int = 48,
        # Model
        distribution: str = "ZINB",
        stabilization: str = "MAD",
        n_iterations: int = 500,
        early_stopping_rounds: int = 30,
        optuna_nfold: int = 3,
        optuna_max_minutes: int = 20,
        # Optuna
        optuna_n_trials: int = 30,
        optuna_param_dict: dict | None = None,
        # Intervals
        n_samples: int = 1000,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        # Reproducibility
        random_seed: int = 42,
    ):
        self.lookback_lags = lookback_lags or [1, 2, 3, 6, 12, 24]
        self.rolling_windows = rolling_windows or [3, 6, 12, 24]
        self.key_weather_features = key_weather_features or []
        self.weather_rolling_windows = weather_rolling_windows or [6, 24]
        self.weather_trend_top_n = weather_trend_top_n

        self.max_samples = max_samples
        self.train_horizons = train_horizons or [
            1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 42, 48,
        ]
        self.max_horizon = max_horizon

        self.distribution = distribution
        self.stabilization = stabilization
        self.n_iterations = n_iterations
        self.early_stopping_rounds = early_stopping_rounds
        self.optuna_nfold = optuna_nfold
        self.optuna_max_minutes = optuna_max_minutes

        self.optuna_n_trials = optuna_n_trials
        self.optuna_param_dict = optuna_param_dict or {
            "eta": ["float", {"low": 0.01, "high": 0.3, "log": True}],
            "max_depth": ["int", {"low": 2, "high": 6, "log": False}],
            "min_child_weight": ["float", {"low": 0.1, "high": 10, "log": True}],
            "subsample": ["float", {"low": 0.5, "high": 1.0, "log": False}],
            "colsample_bytree": ["float", {"low": 0.3, "high": 1.0, "log": False}],
            "gamma": ["float", {"low": 1e-4, "high": 5.0, "log": True}],
        }

        self.n_samples = n_samples
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.random_seed = random_seed

        # Fitted state
        self.model_: Optional[object] = None
        self.feature_columns_: Optional[list[str]] = None
        self.available_weather_: Optional[list[str]] = None
        self.all_weather_features_: Optional[list[str]] = None
        self.n_cpu_: int = multiprocessing.cpu_count()

    # ------------------------------------------------------------------ #
    #  from_config                                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(
        cls, config_path: str | Path = "xgboostlss.yaml",
    ) -> "XGBoostLSSForecaster":
        """Instantiate an XGBoostLSSForecaster from a YAML configuration file."""
        cfg = load_config(config_path)
        feat_cfg = cfg.get("features", {})
        td_cfg = cfg.get("training_data", {})
        model_cfg = cfg.get("model", {})
        optuna_cfg = cfg.get("optuna", {})
        interval_cfg = cfg.get("intervals", {})

        return cls(
            # Feature engineering
            lookback_lags=feat_cfg.get("lookback_lags"),
            rolling_windows=feat_cfg.get("rolling_windows"),
            key_weather_features=feat_cfg.get("key_weather_features"),
            weather_rolling_windows=feat_cfg.get("weather_rolling_windows"),
            weather_trend_top_n=feat_cfg.get("weather_trend_top_n", 10),
            # Training data
            max_samples=td_cfg.get("max_samples", 600_000),
            train_horizons=td_cfg.get("train_horizons"),
            max_horizon=td_cfg.get("max_horizon", 48),
            # Model
            distribution=model_cfg.get("distribution", "ZINB"),
            stabilization=model_cfg.get("stabilization", "MAD"),
            n_iterations=model_cfg.get("n_iterations", 500),
            early_stopping_rounds=model_cfg.get("early_stopping_rounds", 30),
            optuna_nfold=model_cfg.get("optuna_nfold", 3),
            optuna_max_minutes=model_cfg.get("optuna_max_minutes", 20),
            # Optuna
            optuna_n_trials=optuna_cfg.get("n_trials", 30),
            optuna_param_dict=optuna_cfg.get("param_dict"),
            # Intervals
            n_samples=interval_cfg.get("n_samples", 1000),
            lower_quantile=interval_cfg.get("lower_quantile", 0.025),
            upper_quantile=interval_cfg.get("upper_quantile", 0.975),
            # Seed
            random_seed=cfg.get("random_seed", 42),
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _resolve_weather(self, ds) -> tuple[list[str], list[str]]:
        """Intersect configured key weather features with the dataset."""
        all_wf = list(ds.feature.values) if "feature" in ds.dims else []
        available = [f for f in self.key_weather_features if f in all_wf]
        return available, all_wf

    def _make_distribution(self):
        """Instantiate the XGBoostLSS distribution object."""
        from xgboostlss.distributions.ZINB import ZINB

        return ZINB(
            stabilization=self.stabilization,
            response_fn_total_count="relu",
            response_fn_probs="sigmoid",
            loss_fn="nll",
        )

    # ------------------------------------------------------------------ #
    #  fit                                                                 #
    # ------------------------------------------------------------------ #
    def fit(self, ds) -> "XGBoostLSSForecaster":
        """
        Build features and train the XGBoostLSS model.

        Parameters
        ----------
        ds : xr.Dataset
            Training dataset with dims (timestamp, location, feature).
        """
        from xgboostlss.model import XGBoostLSS
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        available_weather, all_weather_features = self._resolve_weather(ds)
        self.available_weather_ = available_weather
        self.all_weather_features_ = all_weather_features
        print(f"  Weather features matched: {len(available_weather)}")

        # Build training data
        train_df = xgblss_build_train(
            ds,
            available_weather,
            all_weather_features,
            lookback_lags=self.lookback_lags,
            rolling_windows=self.rolling_windows,
            weather_rolling_windows=self.weather_rolling_windows,
            weather_trend_top_n=self.weather_trend_top_n,
            train_horizons=self.train_horizons,
            max_samples=self.max_samples,
            random_seed=self.random_seed,
        )

        fc = [c for c in train_df.columns if c != "target"]
        self.feature_columns_ = fc

        X = train_df[fc].values
        y = train_df["target"].values.reshape(-1, 1)

        print(f"\n  Training ZINB XGBoostLSS: {len(X):,} samples, {len(fc)} features")
        dtrain = xgb.DMatrix(X, label=y, nthread=self.n_cpu_, feature_names=fc)

        dist = self._make_distribution()
        model = XGBoostLSS(dist)

        print(f"  Optuna: {self.optuna_n_trials} trials, "
              f"{self.n_iterations} max rounds...")
        opt_params = model.hyper_opt(
            self.optuna_param_dict,
            dtrain,
            num_boost_round=self.n_iterations,
            nfold=self.optuna_nfold,
            early_stopping_rounds=self.early_stopping_rounds,
            max_minutes=self.optuna_max_minutes,
            n_trials=self.optuna_n_trials,
            silence=True,
        )

        n_rounds = opt_params.get("opt_rounds", self.n_iterations)
        print(f"  Best: eta={opt_params.get('eta', 0):.4f}, "
              f"depth={opt_params.get('max_depth', '?')}, rounds={n_rounds}")

        model.train(opt_params, dtrain, num_boost_round=n_rounds)
        print("  Trained!")

        self.model_ = model
        return self

    # ------------------------------------------------------------------ #
    #  predict                                                             #
    # ------------------------------------------------------------------ #
    def predict(
        self,
        ds_context,
        timestamps,
        return_intervals: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecasts for the given timestamps.

        Parameters
        ----------
        ds_context : xr.Dataset
            Historical data up to the forecast origin.
        timestamps : array-like of datetime
            Target timestamps to forecast (up to max_horizon steps).
        return_intervals : bool
            Whether to include lower/upper prediction interval columns.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: timestamp, location, pred,
            [lower, upper].
        """
        if self.model_ is None:
            raise ValueError("Model has not been fitted yet.")

        timestamps = pd.to_datetime(timestamps)

        forecast_df = xgblss_build_forecast(
            ds_context,
            timestamps,
            self.available_weather_,
            self.all_weather_features_,
            lookback_lags=self.lookback_lags,
            rolling_windows=self.rolling_windows,
            weather_rolling_windows=self.weather_rolling_windows,
            weather_trend_top_n=self.weather_trend_top_n,
        )

        fc = self.feature_columns_
        X = forecast_df[fc].values
        dt = xgb.DMatrix(X, nthread=self.n_cpu_, feature_names=fc)

        # Point predictions: mean of distributional samples
        samples = self.model_.predict(
            dt, pred_type="samples", n_samples=self.n_samples,
            seed=self.random_seed,
        )
        pred = np.clip(samples.mean(axis=1).values, 0, None)

        result = pd.DataFrame({
            "timestamp": forecast_df["timestamp"].values,
            "location": forecast_df["location"].values,
            "pred": pred,
        })

        if return_intervals:
            quantiles = self.model_.predict(
                dt, pred_type="quantiles", n_samples=self.n_samples,
                quantiles=[self.lower_quantile, self.upper_quantile],
            )
            lo = np.clip(quantiles.iloc[:, 0].values, 0, None)
            hi = np.clip(quantiles.iloc[:, 1].values, 0, None)
            # Ensure consistency: lower <= pred <= upper
            lo = np.minimum(lo, pred)
            hi = np.maximum(hi, pred)
            result["lower"] = lo
            result["upper"] = hi

        return result
