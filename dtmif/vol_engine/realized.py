"""Realized volatility estimators (Yang–Zhang, close-to-close, EWMA)."""

import numpy as np
import pandas as pd


def yang_zhang_vol(
    ohlcv: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Yang–Zhang realized vol from OHLC (overnight + Rogers–Satchell + open–close mix)."""
    log_ho = np.log(ohlcv["high"] / ohlcv["open"])
    log_lo = np.log(ohlcv["low"] / ohlcv["open"])
    log_co = np.log(ohlcv["close"] / ohlcv["open"])
    log_oc = np.log(ohlcv["open"] / ohlcv["close"].shift(1))
    log_cc = np.log(ohlcv["close"] / ohlcv["close"].shift(1))

    sigma_overnight = log_oc.rolling(window).var()

    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_rs = rs.rolling(window).mean()

    sigma_oc = log_co.rolling(window).var()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    yz_var = sigma_overnight + k * sigma_oc + (1 - k) * sigma_rs
    yz_vol = np.sqrt(yz_var.clip(lower=0))

    if annualize:
        yz_vol = yz_vol * np.sqrt(trading_days)

    yz_vol.name = f"yz_vol_{window}d"
    return yz_vol


def close_to_close_vol(
    close: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """Standard close-to-close historical vol for comparison."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window).std()
    if annualize:
        vol *= np.sqrt(trading_days)
    vol.name = f"hv_{window}d"
    return vol


def ewma_vol(
    close: pd.Series,
    lam: float = 0.94,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """EWMA vol, RiskMetrics λ=0.94."""
    log_ret = np.log(close / close.shift(1)).dropna()
    r = log_ret.values
    n = len(r)
    ev = np.empty(n)
    ev[0] = np.var(r[:20]) if n >= 20 else r[0] ** 2
    for t in range(1, n):
        ev[t] = lam * ev[t - 1] + (1 - lam) * r[t - 1] ** 2
    s = pd.Series(np.sqrt(ev), index=log_ret.index, name="ewma_vol")
    if annualize:
        s *= np.sqrt(trading_days)
    return s
