"""Market data loaders (SPY OHLCV, VIX)."""

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

try:
    import pandas_datareader.data as web
    PDR_OK = True
except ImportError:
    PDR_OK = False


def load_spy_ohlcv(start: str = "2019-01-01", end: str = None) -> pd.DataFrame:
    """SPY daily OHLCV (adjusted)."""
    if not YFINANCE_OK:
        raise ImportError("pip install yfinance")
    ticker = yf.Ticker("SPY")
    df = ticker.history(start=start, end=end)[["Open", "High", "Low", "Close", "Volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.dropna()
    print(f"Loaded SPY OHLCV: {len(df)} rows ({df.index[0].date()}–{df.index[-1].date()})")
    return df


def load_vix(start: str = "2019-01-01") -> pd.Series:
    """VIX daily close from FRED (percentage points → decimal)."""
    if not PDR_OK:
        raise ImportError("pip install pandas-datareader")

    vix = web.DataReader("VIXCLS", "fred", start=start)
    vix.columns = ["vix_raw"]
    vix["vix"] = vix["vix_raw"] / 100.0
    vix.index = pd.to_datetime(vix.index)
    vix = vix["vix"].dropna()
    print(f"Loaded VIX: {len(vix)} rows ({vix.index[0].date()}–{vix.index[-1].date()})")
    return vix
