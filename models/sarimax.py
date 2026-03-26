import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def safe_fit_sarimax(y, order=(1, 0, 1), seasonal_order=None):
    
    y = np.asarray(y, dtype=float).flatten()

    if len(y) < 8 or np.allclose(y, y[0]):
        return None
    try:
        model = SARIMAX(y, 
                        order=order,
                        seasonal_order=seasonal_order if seasonal_order is not None else (0, 0, 0, 0), 
                        enforce_stationarity=False, 
                        enforce_invertibility=False)
        result = model.fit(disp=False)
        return result
    except Exception as e:
        print(f"  Warning: SARIMAX fit failed - {str(e)[:120]}")
        return None
    
class CountySARIMAX:
    def __init__(self, order=(1, 0, 1), seasonal_order=None, clip_nonnegative=True):
        self.order = order
        self.seasonal_order = seasonal_order
        self.clip_nonnegative = clip_nonnegative
        self.models = {}
        self.locations_ = None

    def fit(self, ds):
        """
        ds: xarray Dataset with variable 'out' and dimension 'location'
        """
        locations = list(ds.location.values)
        self.locations_ = [str(loc) for loc in locations]
        self.models = {}

        for loc in locations:
            loc_str = str(loc)
            y_train = ds.out.sel(location=loc).values.astype(float).flatten()
            fitted = safe_fit_sarimax(
                y_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
            )
            self.models[loc_str] = fitted

        return self

    def predict(self, timestamps, locations=None, return_intervals=False, alpha=0.05):
        if locations is None:
            if self.locations_ is None:
                raise ValueError("Model has not been fitted yet.")
            locations = self.locations_

        timestamps = pd.to_datetime(timestamps)
        n_steps = len(timestamps)

        rows = []
        for loc in locations:
            loc_str = str(loc)
            model = self.models.get(loc_str, None)

            if model is None:
                pred = np.zeros(n_steps, dtype=float)
                lower = np.zeros(n_steps, dtype=float)
                upper = np.zeros(n_steps, dtype=float)
            else:
                try:
                    if return_intervals:
                        fc = model.get_forecast(steps=n_steps)
                        pred = np.asarray(fc.predicted_mean, dtype=float)

                        ci = fc.conf_int(alpha=alpha)
                        ci = np.asarray(ci, dtype=float)

                        lower = ci[:, 0]
                        upper = ci[:, 1]
                    else:
                        pred = np.asarray(model.forecast(steps=n_steps), dtype=float)
                        lower = upper = None
                except Exception:
                    pred = np.zeros(n_steps, dtype=float)
                    lower = np.zeros(n_steps, dtype=float)
                    upper = np.zeros(n_steps, dtype=float)

            if self.clip_nonnegative:
                pred = np.clip(pred, 0, None)
                if return_intervals:
                    lower = np.clip(lower, 0, None)
                    upper = np.clip(upper, 0, None)

            data = {
                "timestamp": timestamps,
                "location": loc_str,
                "pred": pred,
            }
            if return_intervals:
                data["lower"] = lower
                data["upper"] = upper

            rows.append(pd.DataFrame(data))

        return pd.concat(rows, ignore_index=True)