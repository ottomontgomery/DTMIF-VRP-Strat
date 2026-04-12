"""P&L with theta–gamma decomposition."""

import numpy as np
import pandas as pd

from dtmif.backtest.config import RISK_FREE_RATE, T_OPTION


def compute_pnl_v2(
    signals: pd.Series,
    sizes: pd.Series,
    realized_vol: pd.Series,
    implied_vol: pd.Series,
    spot: pd.Series,
    r: float = RISK_FREE_RATE,
    T: float = T_OPTION,
    use_theta_gamma: bool = True,
) -> pd.DataFrame:
    """Daily P&L; theta–gamma path when ``use_theta_gamma``, else vega approximation."""
    sig_lag = signals.shift(1)
    size_lag = sizes.shift(1).fillna(0)

    log_ret = np.log(spot / spot.shift(1))

    if use_theta_gamma:
        iv_daily_var = (implied_vol**2) / 252.0
        rv_daily_var = log_ret**2

        sigma_iv = implied_vol.clip(lower=0.01)
        dollar_gamma = spot / (sigma_iv * np.sqrt(T) * np.sqrt(2 * np.pi))

        theta_term = dollar_gamma * iv_daily_var

        pnl_unit = sig_lag * (rv_daily_var - iv_daily_var) * dollar_gamma

        pnl = pnl_unit * size_lag / 100.0

        result = pd.DataFrame(
            {
                "pnl": pnl,
                "theta_pnl": sig_lag * (-1) * theta_term * size_lag / 100.0,
                "gamma_pnl": sig_lag * (-1) * (-rv_daily_var) * dollar_gamma * size_lag / 100.0,
                "position": sig_lag,
                "size": size_lag,
            }
        )

    else:
        vol_surprise = realized_vol - implied_vol
        vega_atm = spot * np.sqrt(T / (2 * np.pi))
        pnl = sig_lag * size_lag * vega_atm * vol_surprise / 100.0

        result = pd.DataFrame(
            {
                "pnl": pnl,
                "theta_pnl": pd.Series(np.nan, index=pnl.index),
                "gamma_pnl": pd.Series(np.nan, index=pnl.index),
                "position": sig_lag,
                "size": size_lag,
            }
        )

    return result.dropna(subset=["pnl"])
