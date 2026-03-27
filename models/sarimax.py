"""
models/sarimax.py
=================
County-level SARIMAX forecaster with optional exogenous regressors.

SARIMAX = Seasonal AutoRegressive Integrated Moving Average with eXogenous
variables.  This model fits an independent SARIMAX process per county.  When
exogenous features are configured (weather variables, temporal encodings), the
model conditions its forecasts on these external signals — leveraging the
full "X" capability of the SARIMAX framework.

Exogenous feature support
-------------------------
SARIMAX requires that the same set of exogenous variables is available for
**both** the training (historical) period **and** the future (forecast)
period.  This limits us to:

- **Weather variables**: provided in the dataset for both train and test.
- **Temporal features**: deterministic (hour, day-of-week, month — cyclic
  encodings), computable for any timestamp.
- **Derived features**: e.g. wind speed from u10/v10.

Outage-derived features (lags, rolling stats) cannot be used here because
they are unknown for the future horizon.  SARIMAX handles its own
autoregressive structure internally via the (p, d, q) and (P, D, Q, s)
orders.

Feature construction is delegated to ``utils.feature_engineering.build_sarimax_exog``
so that the feature set is centralised and reproducible.

Configuration
-------------
Loaded from ``configs/sarimax.yaml`` via the ``from_config`` class method.
The ``exogenous`` section controls which features are used.  Setting all
exogenous options to null/false reduces the model to a plain SARIMA.

Reference: demo.ipynb — Cell 21 (safe_fit_sarimax baseline)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.config import load_config
from utils.feature_engineering import build_sarimax_exog


# =========================================================================== #
#  Helper: safely fit a single SARIMAX model                                   #
# =========================================================================== #

def _safe_fit_sarimax(
    y: np.ndarray,
    exog: np.ndarray | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] | None = None,
):
    """
    Fit a SARIMAX model with graceful error handling.

    Parameters
    ----------
    y : 1-D array
        Endogenous outage time series.
    exog : 2-D array or None
        Exogenous regressors aligned with *y*.  Shape (T, n_features).
        Pass ``None`` for a plain SARIMA fit (no exogenous variables).
    order : tuple
        ARIMA (p, d, q) order.
    seasonal_order : tuple or None
        Seasonal (P, D, Q, s) order.

    Returns
    -------
    statsmodels SARIMAXResults or None
        ``None`` when the series is too short, constant, or the optimiser
        fails to converge — the caller falls back to a zero forecast.
    """
    y = np.asarray(y, dtype=float).flatten()

    # Guard: not enough data or zero variance
    if len(y) < 8 or np.allclose(y, y[0]):
        return None

    # Guard: if exog is provided, ensure lengths match
    if exog is not None:
        exog = np.asarray(exog, dtype=float)
        if exog.shape[0] != len(y):
            print(f"  Warning: exog length ({exog.shape[0]}) != y length ({len(y)}), skipping")
            return None

    try:
        model = SARIMAX(
            y,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order if seasonal_order is not None else (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        result = model.fit(disp=False)
        return result
    except Exception as e:
        print(f"  Warning: SARIMAX fit failed — {str(e)[:120]}")
        return None


# =========================================================================== #
#  CountySARIMAX forecaster                                                    #
# =========================================================================== #

class CountySARIMAX:
    """
    Fits one SARIMAX model per county and produces forecasts + PIs.

    When exogenous features are configured (via the YAML config), the model
    extracts weather and temporal features at both train and predict time
    using the shared ``build_sarimax_exog`` helper.

    Parameters
    ----------
    order : tuple[int, int, int]
        ARIMA (p, d, q) order.
    seasonal_order : tuple or None
        Seasonal (P, D, Q, s) order.  ``None`` disables seasonality.
    clip_nonnegative : bool
        If True, clip predictions and intervals to [0, ∞).
    alpha : float
        Significance level for prediction intervals (default 0.05 → 95%).
    weather_features : list[str] or None
        Weather variables to include as exogenous regressors.
        ``None`` or ``[]`` means no weather features.
    include_temporal : bool
        Whether to include deterministic temporal encodings.
    include_wind_speed : bool
        Whether to derive and include wind speed from u10/v10.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        clip_nonnegative: bool = True,
        alpha: float = 0.05,
        weather_features: list[str] | None = None,
        include_temporal: bool = True,
        include_wind_speed: bool = True,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.clip_nonnegative = clip_nonnegative
        self.alpha = alpha

        # Exogenous feature settings
        self.weather_features = weather_features or []
        self.include_temporal = include_temporal
        self.include_wind_speed = include_wind_speed

        # Internal state
        self.models: dict[str, object] = {}
        self.locations_: Optional[list[str]] = None
        self.n_exog_features_: int = 0  # set during fit, for consistency check

    @property
    def has_exog(self) -> bool:
        """Whether this model is configured with exogenous regressors."""
        return bool(self.weather_features) or self.include_temporal

    # ------------------------------------------------------------------ #
    #  Factory: build from YAML config                                     #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(cls, config_path: str | Path = "sarimax.yaml") -> "CountySARIMAX":
        """
        Instantiate a CountySARIMAX from a YAML configuration file.

        The ``exogenous`` section of the YAML controls which features are
        used.  Example::

            exogenous:
              weather_features: [gust, t2m, prate, u10, v10]
              include_temporal: true
              include_wind_speed: true
        """
        cfg = load_config(config_path)
        model_cfg = cfg.get("model", {})
        interval_cfg = cfg.get("intervals", {})
        exog_cfg = cfg.get("exogenous", {})

        order = tuple(model_cfg.get("order", [1, 0, 1]))
        raw_seasonal = model_cfg.get("seasonal_order", None)
        seasonal_order = tuple(raw_seasonal) if raw_seasonal is not None else None

        # Parse exogenous settings
        weather_features = exog_cfg.get("weather_features", None)
        # Normalise: null / empty list → no weather features
        if weather_features is None:
            weather_features = []

        return cls(
            order=order,
            seasonal_order=seasonal_order,
            clip_nonnegative=model_cfg.get("clip_nonnegative", True),
            alpha=interval_cfg.get("alpha", 0.05),
            weather_features=weather_features,
            include_temporal=exog_cfg.get("include_temporal", True),
            include_wind_speed=exog_cfg.get("include_wind_speed", True),
        )

    # ------------------------------------------------------------------ #
    #  Exogenous feature extraction (delegates to shared module)           #
    # ------------------------------------------------------------------ #

    def _build_exog(
        self,
        ds,
        location: str,
        timestamps: pd.DatetimeIndex | None = None,
    ) -> np.ndarray | None:
        """
        Build exogenous features for one county.

        Delegates to ``utils.feature_engineering.build_sarimax_exog``,
        forwarding the weather/temporal/wind-speed settings from the YAML.

        Returns None if no exogenous features are configured.
        """
        if not self.has_exog:
            return None

        return build_sarimax_exog(
            ds=ds,
            location=location,
            timestamps=timestamps,
            weather_features=self.weather_features if self.weather_features else None,
            include_temporal=self.include_temporal,
            include_wind_speed=self.include_wind_speed,
        )

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, ds) -> "CountySARIMAX":
        """
        Fit one SARIMAX model per county in the xarray Dataset *ds*.

        If exogenous features are configured, the weather and temporal
        regressors are extracted from the dataset and passed to each
        per-county SARIMAX fit.

        Parameters
        ----------
        ds : xr.Dataset
            Must have variable ``out`` (and ``weather`` if using exogenous
            weather features) with dimensions (timestamp, location).
        """
        locations = list(ds.location.values)
        self.locations_ = [str(loc) for loc in locations]
        self.models = {}

        if self.has_exog:
            feature_desc = []
            if self.weather_features:
                feature_desc.append(f"{len(self.weather_features)} weather vars")
            if self.include_temporal:
                feature_desc.append("temporal")
            if self.include_wind_speed and "u10" in self.weather_features and "v10" in self.weather_features:
                feature_desc.append("wind_speed")
            print(f"  [SARIMAX] Exogenous features: {', '.join(feature_desc)}")

        for loc in locations:
            loc_str = str(loc)
            y_train = ds.out.sel(location=loc).values.astype(float).flatten()

            # Build exogenous matrix for this county's training period
            exog_train = self._build_exog(ds, location=loc_str)

            if exog_train is not None:
                # Record the feature count for consistency check at predict time
                self.n_exog_features_ = exog_train.shape[1]

            fitted = _safe_fit_sarimax(
                y_train,
                exog=exog_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self.models[loc_str] = fitted

        return self

    # ------------------------------------------------------------------ #
    #  Prediction                                                          #
    # ------------------------------------------------------------------ #

    def predict(
        self,
        timestamps,
        locations: list[str] | None = None,
        return_intervals: bool = False,
        alpha: float | None = None,
        ds_future=None,
    ) -> pd.DataFrame:
        """
        Produce forecasts for the given *timestamps*.

        Parameters
        ----------
        timestamps : array-like of datetime
            Forecast target timestamps.
        locations : list[str], optional
            Counties to predict.  Defaults to all counties seen during fit.
        return_intervals : bool
            Whether to include ``lower`` and ``upper`` columns.
        alpha : float, optional
            Override the significance level for PIs.
        ds_future : xr.Dataset, optional
            Dataset covering the forecast period.  **Required when the model
            is configured with exogenous weather features** — the weather
            data for the future timestamps is extracted from this dataset.
            When only temporal features are used (no weather), this can be
            ``None`` because temporal features are deterministic.

        Returns
        -------
        pd.DataFrame
            Long-format with columns: timestamp, location, pred,
            [lower, upper].

        Raises
        ------
        ValueError
            If exogenous weather features are configured but *ds_future*
            is not provided.
        """
        if locations is None:
            if self.locations_ is None:
                raise ValueError("Model has not been fitted yet.")
            locations = self.locations_
        if alpha is None:
            alpha = self.alpha

        timestamps = pd.to_datetime(timestamps)
        n_steps = len(timestamps)

        # Validate: if we used weather exog during training, we need future
        # weather data to forecast.
        if self.weather_features and ds_future is None:
            raise ValueError(
                "This model was trained with exogenous weather features "
                f"({self.weather_features}), but no ds_future was provided "
                "for the prediction period.  Pass ds_future= so that future "
                "weather data can be extracted."
            )

        rows: list[pd.DataFrame] = []
        for loc in locations:
            loc_str = str(loc)
            model = self.models.get(loc_str, None)

            # Build exogenous features for the prediction period
            exog_future = None
            if self.has_exog:
                if ds_future is not None:
                    exog_future = self._build_exog(
                        ds_future, location=loc_str, timestamps=timestamps,
                    )
                elif self.include_temporal and not self.weather_features:
                    # Temporal-only mode: we can build features from timestamps alone
                    from utils.feature_engineering import build_temporal_features
                    exog_future = build_temporal_features(timestamps).values

            if model is None:
                # Fallback: zero forecast when the model could not be fitted
                pred = np.zeros(n_steps, dtype=float)
                lower = np.zeros(n_steps, dtype=float)
                upper = np.zeros(n_steps, dtype=float)
            else:
                try:
                    if return_intervals:
                        fc = model.get_forecast(steps=n_steps, exog=exog_future)
                        pred = np.asarray(fc.predicted_mean, dtype=float)
                        ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
                        lower, upper = ci[:, 0], ci[:, 1]
                    else:
                        pred = np.asarray(
                            model.forecast(steps=n_steps, exog=exog_future),
                            dtype=float,
                        )
                        lower = upper = None
                except Exception as e:
                    print(f"  Warning: SARIMAX forecast failed for {loc_str} — {str(e)[:80]}")
                    pred = np.zeros(n_steps, dtype=float)
                    lower = np.zeros(n_steps, dtype=float)
                    upper = np.zeros(n_steps, dtype=float)

            # Outages are non-negative by definition
            if self.clip_nonnegative:
                pred = np.clip(pred, 0, None)
                if return_intervals and lower is not None:
                    lower = np.clip(lower, 0, None)
                    upper = np.clip(upper, 0, None)

            data: dict = {
                "timestamp": timestamps,
                "location": loc_str,
                "pred": pred,
            }
            if return_intervals and lower is not None:
                data["lower"] = lower
                data["upper"] = upper

            rows.append(pd.DataFrame(data))

        return pd.concat(rows, ignore_index=True)
