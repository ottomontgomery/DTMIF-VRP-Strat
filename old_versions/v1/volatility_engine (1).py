"""
volatility_engine.py
=====================
Volatility Engine for SPY
  - 20-day and 60-day rolling Historical Volatility (HV)
  - EWMA Volatility (RiskMetrics, lambda = 0.94)
  - Black-Scholes Implied Volatility via brentq root-finding

Author  : Quant Finance Project
Requires: numpy, pandas, scipy, yfinance
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
import warnings

# Suppress yfinance deprecation noise
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("yfinance not installed. Install with: pip install yfinance")


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_spy_prices(start: str = "2019-01-01", end: str = None) -> pd.DataFrame:
    """
    Download SPY daily adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    start : str  - Start date in 'YYYY-MM-DD' format (default 5 years back)
    end   : str  - End date  (default: today)

    Returns
    -------
    pd.DataFrame with columns: ['spy_close']
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Run: pip install yfinance")

    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start, end=end)
    df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone
    df = df[["Close"]].rename(columns={"Close": "spy_close"})
    df = df.dropna()
    print(f"Loaded {len(df)} rows of SPY data ({df.index[0].date()} — {df.index[-1].date()})")
    return df


def load_spy_options_snapshot(symbol: str = "SPY") -> dict:
    """
    Download SPY near-dated options chain from Yahoo Finance.

    Returns a dict with:
        'expirations'  : list of available expiry strings
        'calls'        : pd.DataFrame of call options for nearest expiry
        'puts'         : pd.DataFrame of put options for nearest expiry
        'spot'         : current underlying price
        'expiry_used'  : the expiry date selected

    Notes
    -----
    - Yahoo Finance returns only the current snapshot; for backtesting
      historical IV you need a paid data source (see data_sources.md).
    - Columns returned: contractSymbol, strike, lastPrice, bid, ask,
      impliedVolatility, volume, openInterest, inTheMoney
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required.")

    ticker = yf.Ticker(symbol)
    spot = ticker.fast_info["lastPrice"]
    expirations = ticker.options

    if len(expirations) == 0:
        raise ValueError(f"No options data found for {symbol}.")

    # Use the nearest expiry that is ≥7 days away (avoid same-week pinning)
    today = pd.Timestamp.today().normalize()
    valid = [e for e in expirations if (pd.Timestamp(e) - today).days >= 7]
    expiry = valid[0] if valid else expirations[0]

    chain = ticker.option_chain(expiry)

    return {
        "expirations": list(expirations),
        "calls": chain.calls,
        "puts":  chain.puts,
        "spot":  spot,
        "expiry_used": expiry,
    }


# ============================================================
# 2. LOG RETURNS
# ============================================================

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily log returns: r_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series  - Price series (e.g. adjusted close)

    Returns
    -------
    pd.Series of log returns (NaN on the first observation dropped)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    log_ret.name = "log_return"
    return log_ret


# ============================================================
# 3. ROLLING HISTORICAL VOLATILITY
# ============================================================

def rolling_historical_vol(
    log_returns: pd.Series,
    window: int,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute rolling historical volatility (close-to-close).

    Formula: HV_t = std(r_{t-w+1}, ..., r_t)  [× sqrt(252) if annualised]

    Parameters
    ----------
    log_returns   : pd.Series  - Daily log returns
    window        : int        - Rolling window in trading days
    annualize     : bool       - Multiply by sqrt(trading_days) if True
    trading_days  : int        - Convention for annualisation (default 252)

    Returns
    -------
    pd.Series named 'hv_{window}d'
    """
    vol = log_returns.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(trading_days)
    vol.name = f"hv_{window}d"
    return vol


# ============================================================
# 4. EWMA VOLATILITY  (RiskMetrics λ = 0.94)
# ============================================================

def ewma_volatility(
    log_returns: pd.Series,
    lam: float = 0.94,
    annualize: bool = True,
    trading_days: int = 252,
) -> pd.Series:
    """
    Compute EWMA (Exponentially Weighted Moving Average) volatility.

    RiskMetrics recursive update (J.P. Morgan, 1994):
        σ²_t = λ · σ²_{t-1} + (1-λ) · r²_{t-1}

    A lower λ makes the estimate react faster to recent returns;
    λ = 0.94 is the RiskMetrics standard for daily data.

    Parameters
    ----------
    log_returns  : pd.Series  - Daily log returns
    lam          : float      - Decay factor (0 < λ < 1, default 0.94)
    annualize    : bool       - Multiply by sqrt(trading_days) if True
    trading_days : int        - Convention for annualisation (default 252)

    Returns
    -------
    pd.Series named 'ewma_vol'
    """
    r = log_returns.values
    n = len(r)
    ewma_var = np.empty(n)

    # Seed: use variance of first min(20, n) observations
    seed_window = min(20, n)
    ewma_var[0] = np.var(r[:seed_window]) if seed_window > 1 else r[0] ** 2

    for t in range(1, n):
        ewma_var[t] = lam * ewma_var[t - 1] + (1 - lam) * r[t - 1] ** 2

    ewma_vol = pd.Series(
        np.sqrt(ewma_var), index=log_returns.index, name="ewma_vol"
    )

    if annualize:
        ewma_vol = ewma_vol * np.sqrt(trading_days)

    return ewma_vol


# ============================================================
# 5. BLACK-SCHOLES PRICING
# ============================================================

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes price for a European option (no dividends).

    Parameters
    ----------
    S           : float  - Current underlying price
    K           : float  - Strike price
    T           : float  - Time to expiration in years (e.g. 30/365)
    r           : float  - Continuous risk-free rate (annualised, e.g. 0.05)
    sigma       : float  - Volatility (annualised, e.g. 0.20)
    option_type : str    - 'call' or 'put'

    Returns
    -------
    float: theoretical option price
    """
    if T <= 0:
        # At expiry: return intrinsic value
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# ============================================================
# 6. IMPLIED VOLATILITY (Brent's Method)
# ============================================================

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    vol_low: float = 1e-6,
    vol_high: float = 10.0,
) -> float:
    """
    Extract Black-Scholes implied volatility using Brent's root-finding method.

    We solve: BS_price(σ) − market_price = 0  for σ ∈ [vol_low, vol_high]

    Parameters
    ----------
    market_price : float  - Observed mid-market option price
    S            : float  - Underlying price
    K            : float  - Strike price
    T            : float  - Time to expiration in years
    r            : float  - Continuous risk-free rate
    option_type  : str    - 'call' or 'put'
    vol_low      : float  - Lower bracket for brentq (default 1e-6)
    vol_high     : float  - Upper bracket for brentq (default 10.0 = 1000% vol)

    Returns
    -------
    float: implied volatility, or np.nan if no solution exists
    """
    # --- Arbitrage / validity checks ---
    forward = S * np.exp(r * T)
    pv_K    = K * np.exp(-r * T)

    if option_type == "call":
        intrinsic = max(S - pv_K, 0.0)
    else:
        intrinsic = max(pv_K - S, 0.0)

    if market_price < intrinsic - 1e-6:
        # Price below intrinsic: arbitrage — return NaN
        return np.nan

    if market_price <= 0:
        return np.nan

    if T <= 0:
        return np.nan

    # --- Root-finding ---
    objective = lambda sigma: (
        black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    )

    # Check bracket validity
    try:
        f_low  = objective(vol_low)
        f_high = objective(vol_high)
    except Exception:
        return np.nan

    if f_low * f_high > 0:
        # No sign change in bracket — brentq will fail
        return np.nan

    try:
        iv = brentq(objective, vol_low, vol_high, maxiter=1000, xtol=1e-8, rtol=1e-8)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def compute_iv_series(
    options_df: pd.DataFrame,
    r: float = 0.05,
    price_col: str = "market_price",
    S_col: str = "S",
    K_col: str = "K",
    T_col: str = "T",
    type_col: str = "option_type",
) -> pd.DataFrame:
    """
    Vectorised IV computation over a DataFrame of option quotes.

    Expected input columns (configurable via arguments):
        market_price  - observed mid-market price
        S             - underlying price at quote time
        K             - strike price
        T             - time to expiration in years
        option_type   - 'call' or 'put'

    Returns
    -------
    Input DataFrame with an additional 'implied_vol' column.
    """
    df = options_df.copy()
    df["implied_vol"] = df.apply(
        lambda row: implied_volatility(
            market_price=row[price_col],
            S=row[S_col],
            K=row[K_col],
            T=row[T_col],
            r=r,
            option_type=row.get(type_col, "call"),
        ),
        axis=1,
    )
    return df


# ============================================================
# MAIN — DEMO RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  VOLATILITY ENGINE — SPY DEMO")
    print("=" * 60)

    # ---- 1. Load prices ----
    prices   = load_spy_prices(start="2019-01-01")
    log_ret  = compute_log_returns(prices["spy_close"])

    # ---- 2. Historical volatility ----
    hv_20 = rolling_historical_vol(log_ret, window=20)
    hv_60 = rolling_historical_vol(log_ret, window=60)

    # ---- 3. EWMA volatility ----
    ewma_vol = ewma_volatility(log_ret, lam=0.94)

    # ---- 4. Combine and save ----
    vol_df = pd.DataFrame({
        "spy_close": prices["spy_close"],
        "log_return": log_ret,
        "hv_20d":    hv_20,
        "hv_60d":    hv_60,
        "ewma_vol":  ewma_vol,
    }).dropna()

    vol_df.to_csv("spy_volatility.csv")
    print(f"\nSaved spy_volatility.csv  ({len(vol_df)} rows)")
    print("\nLast 5 rows:")
    print(vol_df[["hv_20d", "hv_60d", "ewma_vol"]].tail(5).round(4))

    # ---- 5. IV round-trip sanity check ----
    print("\n" + "-" * 40)
    print("  IV ROUND-TRIP SANITY CHECK")
    print("-" * 40)
    S, K, T, r = 500.0, 500.0, 30 / 365, 0.05
    for true_vol in [0.15, 0.20, 0.30]:
        price = black_scholes_price(S, K, T, r, sigma=true_vol, option_type="call")
        recovered = implied_volatility(price, S, K, T, r, "call")
        print(f"  True σ={true_vol:.2f}  →  BS price={price:.4f}  →  IV recovered={recovered:.4f}")

    # ---- 6. Current options snapshot (live) ----
    print("\n" + "-" * 40)
    print("  LIVE OPTIONS SNAPSHOT (nearest expiry)")
    print("-" * 40)
    try:
        snap = load_spy_options_snapshot("SPY")
        print(f"  Spot: ${snap['spot']:.2f}  |  Expiry: {snap['expiry_used']}")
        calls = snap["calls"]
        # Compute IV for the first 5 calls using last-trade price
        T_snap = (pd.Timestamp(snap["expiry_used"]) - pd.Timestamp.today()).days / 365
        sample = calls[calls["lastPrice"] > 0.01].head(5).copy()
        sample = sample.rename(columns={"strike": "K", "lastPrice": "market_price"})
        sample["S"] = snap["spot"]
        sample["T"] = T_snap
        sample["option_type"] = "call"
        sample = compute_iv_series(sample, r=0.05)
        print(sample[["K", "market_price", "implied_vol"]].to_string(index=False))
    except Exception as e:
        print(f"  (Options snapshot skipped: {e})")
