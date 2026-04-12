"""HAR-RV (Corsi) forecast."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def har_rv_forecast(
    close: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
    forecast_horizon: int = 1,
) -> pd.Series:
    """HAR-RV rolling OLS forecast from daily, weekly, and monthly realized variance lags."""
    log_ret = np.log(close / close.shift(1)).dropna()
    rv = log_ret**2

    rv_d = rv.shift(1)
    rv_w = rv.rolling(5).mean().shift(1)
    rv_m = rv.rolling(22).mean().shift(1)

    rv_next = rv.shift(-forecast_horizon)

    X = pd.concat([rv_d, rv_w, rv_m], axis=1)
    X.columns = ["rv_d", "rv_w", "rv_m"]
    y = rv_next

    data = pd.concat([X, y.rename("rv_next")], axis=1).dropna()

    n_min = 60
    forecasts = pd.Series(np.nan, index=rv.index)

    X_arr = data[["rv_d", "rv_w", "rv_m"]].values
    y_arr = data["rv_next"].values
    dates = data.index

    for i in range(n_min, len(data)):
        X_train = X_arr[:i]
        y_train = y_arr[:i]
        X_pred = X_arr[i : i + 1]

        reg = LinearRegression().fit(X_train, y_train)
        rv_forecast = max(reg.predict(X_pred)[0], 0)
        forecasts[dates[i]] = rv_forecast

    vol_forecast = np.sqrt(forecasts.clip(lower=0))
    if annualize:
        vol_forecast = vol_forecast * np.sqrt(trading_days)

    vol_forecast.name = "har_vol"
    return vol_forecast
