"""
vol_engine_v2.py
=================
Improved Volatility Engine — v2
  - Yang-Zhang realized vol estimator (uses OHLCV, lower bias than close-to-close)
  - GARCH(1,1) vol forecast (arch library)
  - HAR-RV model (daily + weekly + monthly realized var components)
  - Forecast combination (equal-weight ensemble)
  - Real VIX data via FRED (IV proxy, no circular simulation)

Author  : Quant Finance Project
Requires: numpy, pandas, scipy, arch, yfinance, pandas_datareader
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("arch not installed. Install with: pip install arch")

PDR_OK = False


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_spy_ohlcv(start: str = "2019-01-01", end: str = None) -> pd.DataFrame:
    """
    Load SPY daily OHLCV data (adjusted).
    Required for Yang-Zhang and Rogers-Satchell estimators.
    """
    if not YFINANCE_OK:
        raise ImportError("pip install yfinance")
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start, end=end)[["Open","High","Low","Close","Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = ["open","high","low","close","volume"]
    df = df.dropna()
    print(f"Loaded SPY OHLCV: {len(df)} rows ({df.index[0].date()}–{df.index[-1].date()})")
    return df


def load_vix(start: str = "2019-01-01") -> pd.Series:
    """
    Download CBOE VIX daily close from FRED.

    VIX is the model-free 30-day IV of S&P 500 options.
    Use as the IV proxy instead of simulating IV from EWMA.

    Note: VIX is quoted as an annualised vol percentage (e.g. 20 = 20% vol).
    We convert to decimal for comparability with HV measures.
    """
    if not PDR_OK:
        raise ImportError("pip install pandas_datareader")

    vix = web.DataReader("VIXCLS", "fred", start=start)
    vix.columns = ["vix_raw"]
    vix["vix"] = vix["vix_raw"] / 100.0   # convert to decimal
    vix.index = pd.to_datetime(vix.index)
    vix = vix["vix"].dropna()
    print(f"Loaded VIX: {len(vix)} rows ({vix.index[0].date()}–{vix.index[-1].date()})")
    return vix


# ============================================================
# 2. REALIZED VOL ESTIMATORS
# ============================================================

def yang_zhang_vol(
    ohlcv: pd.DataFrame,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Yang-Zhang (2000) volatility estimator.

    Minimum-variance unbiased estimator using open, high, low, close.
    Handles overnight gaps and intraday volatility separately — typically
    30–40% more efficient than close-to-close standard deviation.

    Components:
        σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS

    where:
        σ²_overnight = var(log(Open_t / Close_{t-1}))   [overnight component]
        σ²_RS        = Rogers-Satchell intraday estimator
        k            = 0.34 / (1.34 + (n+1)/(n-1))     [optimal mixing weight]

    Parameters
    ----------
    ohlcv   : pd.DataFrame with columns open, high, low, close
    window  : int  - Rolling window in trading days
    annualize : bool

    Returns
    -------
    pd.Series named 'yz_vol'
    """
    log_ho = np.log(ohlcv["high"] / ohlcv["open"])
    log_lo = np.log(ohlcv["low"]  / ohlcv["open"])
    log_co = np.log(ohlcv["close"] / ohlcv["open"])
    log_oc = np.log(ohlcv["open"] / ohlcv["close"].shift(1))
    log_cc = np.log(ohlcv["close"] / ohlcv["close"].shift(1))

    # Overnight variance
    sigma_overnight = log_oc.rolling(window).var()

    # Rogers-Satchell intraday estimator (no mean assumption)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    sigma_rs = rs.rolling(window).mean()

    # Open-to-close variance
    sigma_oc = log_co.rolling(window).var()

    # Optimal weight
    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    # Yang-Zhang combined estimate
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
    """EWMA volatility — RiskMetrics λ=0.94 (from v1, unchanged)."""
    log_ret = np.log(close / close.shift(1)).dropna()
    r = log_ret.values
    n = len(r)
    ev = np.empty(n)
    ev[0] = np.var(r[:20]) if n >= 20 else r[0]**2
    for t in range(1, n):
        ev[t] = lam * ev[t-1] + (1 - lam) * r[t-1]**2
    s = pd.Series(np.sqrt(ev), index=log_ret.index, name="ewma_vol")
    if annualize:
        s *= np.sqrt(trading_days)
    return s


# ============================================================
# 3. GARCH(1,1) VOLATILITY FORECAST
# ============================================================

def garch_vol_forecast(
    close: pd.Series,
    p: int = 1,
    q: int = 1,
    annualize: bool = True,
    trading_days: int = 252,
    use_gjr: bool = True,
) -> pd.Series:
    """
    One-step-ahead conditional volatility forecast from GARCH(1,1) or GJR-GARCH.

    GJR-GARCH (Glosten-Jagannathan-Runkle, 1993) is preferred over standard
    GARCH for equity vol because it captures the leverage effect:
    negative returns increase subsequent volatility more than positive returns
    of the same magnitude. For SPY, this is empirically well-documented.

    Parameters
    ----------
    close     : pd.Series  - Adjusted close prices
    p, q      : int        - GARCH lag orders
    use_gjr   : bool       - Use GJR-GARCH (asymmetric) if True, else GARCH
    annualize : bool       - Multiply by sqrt(252)

    Returns
    -------
    pd.Series of one-step-ahead conditional vol forecast, same index as close
    """
    if not ARCH_OK:
        print("WARNING: arch not available. Returning EWMA as fallback.")
        return ewma_vol(close, annualize=annualize)

    log_ret = np.log(close / close.shift(1)).dropna() * 100  # percent scale for arch

    # Fit model
    vol_type = "EGARCH" if use_gjr else "GARCH"
    # Use GJR-GARCH (more robust than EGARCH for forecasting)
    try:
        if use_gjr:
            model = arch_model(log_ret, vol="Garch", p=p, o=1, q=q, dist="normal")
        else:
            model = arch_model(log_ret, vol="Garch", p=p, q=q, dist="normal")
        res = model.fit(disp="off", show_warning=False)
    except Exception as e:
        print(f"GARCH fit failed ({e}). Using EWMA fallback.")
        return ewma_vol(close, annualize=annualize)

    # Extract conditional vol (annualize: divide by 100 back to decimal, then ×√252)
    cond_vol = res.conditional_volatility / 100.0
    if annualize:
        cond_vol = cond_vol * np.sqrt(trading_days)

    cond_vol.name = "gjr_garch_vol"
    return cond_vol


# ============================================================
# 4. HAR-RV MODEL (Corsi 2009)
# ============================================================

def har_rv_forecast(
    close: pd.Series,
    annualize: bool = True,
    trading_days: int = 252,
    forecast_horizon: int = 1,
) -> pd.Series:
    """
    Heterogeneous Autoregressive Realized Variance (HAR-RV) model.
    Corsi (2009) — one of the best performing realized vol forecasting models.

    Specification:
        RV_{t+h} = α + β_d×RV_d_t + β_w×RV_w_t + β_m×RV_m_t + ε

    where:
        RV_d  = daily realized variance (squared log return)
        RV_w  = average of last 5 daily RVs (weekly component)
        RV_m  = average of last 22 daily RVs (monthly component)

    The model captures the long-memory property of volatility using a
    simple linear structure with three overlapping realized var components.

    Parameters
    ----------
    close            : pd.Series  - Adjusted close prices
    forecast_horizon : int        - Forecast steps ahead (default 1 = tomorrow)

    Returns
    -------
    pd.Series: one-step-ahead vol forecast
    """
    from sklearn.linear_model import LinearRegression

    log_ret = np.log(close / close.shift(1)).dropna()
    rv = log_ret**2   # daily realized variance

    # Lagged components
    rv_d = rv.shift(1)             # yesterday's RV
    rv_w = rv.rolling(5).mean().shift(1)   # last week's avg RV
    rv_m = rv.rolling(22).mean().shift(1)  # last month's avg RV

    # Target: next day's RV
    rv_next = rv.shift(-forecast_horizon)

    # Combine into DataFrame
    X = pd.concat([rv_d, rv_w, rv_m], axis=1)
    X.columns = ["rv_d", "rv_w", "rv_m"]
    y = rv_next

    data = pd.concat([X, y.rename("rv_next")], axis=1).dropna()

    # Rolling OLS forecast (60-day expanding window minimum, then rolling 252-day)
    n_min = 60
    forecasts = pd.Series(np.nan, index=rv.index)

    X_arr = data[["rv_d","rv_w","rv_m"]].values
    y_arr = data["rv_next"].values
    dates = data.index

    for i in range(n_min, len(data)):
        # Fit on all data up to i-1 (no look-ahead)
        X_train = X_arr[:i]
        y_train = y_arr[:i]
        X_pred  = X_arr[i:i+1]

        # Simple OLS
        reg = LinearRegression().fit(X_train, y_train)
        rv_forecast = max(reg.predict(X_pred)[0], 0)  # clip to positive
        forecasts[dates[i]] = rv_forecast

    # Convert variance forecast to vol forecast, annualize
    vol_forecast = np.sqrt(forecasts.clip(lower=0))
    if annualize:
        vol_forecast = vol_forecast * np.sqrt(trading_days)

    vol_forecast.name = "har_vol"
    return vol_forecast


# ============================================================
# 5. FORECAST COMBINATION
# ============================================================

def combine_forecasts(
    forecasts: dict,
    weights: dict = None,
) -> pd.Series:
    """
    Combine multiple vol forecasts into an ensemble.

    Equal-weight combination of diverse forecasting models has been shown
    to outperform individual models out-of-sample (Timmermann, 2006).

    Parameters
    ----------
    forecasts : dict  - {name: pd.Series} of vol forecasts
    weights   : dict  - Optional {name: float} weights (must sum to 1).
                        Default is equal-weight.

    Returns
    -------
    pd.Series: weighted combination
    """
    df = pd.DataFrame(forecasts).dropna()
    if weights is None:
        w = {k: 1/len(forecasts) for k in forecasts}
    else:
        total = sum(weights.values())
        w = {k: v/total for k, v in weights.items()}

    combined = sum(df[k] * w[k] for k in forecasts)
    combined.name = "ensemble_vol"
    return combined


# ============================================================
# 6. BLACKS-SCHOLES (unchanged from v1)
# ============================================================

def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type="call",
                vol_lo=1e-6, vol_hi=10.0):
    pv_K = K*np.exp(-r*T)
    intrinsic = max(S - pv_K, 0) if option_type=="call" else max(pv_K - S, 0)
    if market_price < intrinsic - 1e-6 or T <= 0 or market_price <= 0:
        return np.nan
    try:
        f = lambda s: bs_price(S, K, T, r, s, option_type) - market_price
        if f(vol_lo) * f(vol_hi) > 0:
            return np.nan
        return brentq(f, vol_lo, vol_hi, maxiter=1000, xtol=1e-8)
    except Exception:
        return np.nan


# ============================================================
# MAIN — DEMO
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("  VOLATILITY ENGINE v2 — DEMO")
    print("="*60)

    # Load OHLCV
    ohlcv = load_spy_ohlcv(start="2019-01-01")

    # Load VIX
    try:
        vix = load_vix(start="2019-01-01")
    except Exception as e:
        print(f"VIX load failed: {e}")
        vix = None

    # Compute all estimators
    print("\nComputing vol estimators...")
    hv20   = close_to_close_vol(ohlcv["close"], window=20)
    yz20   = yang_zhang_vol(ohlcv, window=20)
    ewma   = ewma_vol(ohlcv["close"])

    print("Fitting GARCH(1,1) / GJR-GARCH...")
    garch  = garch_vol_forecast(ohlcv["close"], use_gjr=True)

    print("Fitting HAR-RV (rolling OLS)...")
    har    = har_rv_forecast(ohlcv["close"])

    # Combine
    forecasts = {k: v for k, v in {
        "ewma": ewma, "garch": garch, "har": har
    }.items() if v is not None}
    ensemble = combine_forecasts(forecasts)

    # Summary table
    result = pd.DataFrame({
        "HV-20":    hv20,
        "YZ-20":    yz20,
        "EWMA":     ewma,
        "GJR-GARCH":garch,
        "HAR-RV":   har,
        "Ensemble": ensemble,
    }).dropna()

    if vix is not None:
        vix_aligned = vix.reindex(result.index, method="ffill")
        result["VIX"] = vix_aligned

    print(f"\nFinal DataFrame: {len(result)} rows")
    print("\nLast 5 rows:")
    print(result.tail(5).round(4))

    result.to_csv("spy_volatility_v2.csv")
    print("\nSaved → spy_volatility_v2.csv")

    # Forecast accuracy (RMSE vs HV-20 as proxy for realized vol)
    from sklearn.metrics import mean_squared_error
    common = result.dropna()
    rv_proxy = common["HV-20"]
    print("\n=== FORECAST RMSE vs HV-20 REALIZED ===")
    for col in ["EWMA", "GJR-GARCH", "HAR-RV", "Ensemble"]:
        if col in common.columns:
            rmse = np.sqrt(mean_squared_error(rv_proxy, common[col]))
            print(f"  {col:<15}: RMSE = {rmse:.6f}")