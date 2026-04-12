"""Black–Scholes price and implied vol (optional utilities)."""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(
    market_price, S, K, T, r, option_type="call", vol_lo=1e-6, vol_hi=10.0
):
    pv_K = K * np.exp(-r * T)
    intrinsic = max(S - pv_K, 0) if option_type == "call" else max(pv_K - S, 0)
    if market_price < intrinsic - 1e-6 or T <= 0 or market_price <= 0:
        return np.nan
    try:

        def f(s):
            return bs_price(S, K, T, r, s, option_type) - market_price

        if f(vol_lo) * f(vol_hi) > 0:
            return np.nan
        return brentq(f, vol_lo, vol_hi, maxiter=1000, xtol=1e-8)
    except Exception:
        return np.nan
