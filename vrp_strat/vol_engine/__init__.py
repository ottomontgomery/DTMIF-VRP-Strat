"""Volatility engine: realized vol, GARCH, HAR-RV, ensemble, loaders."""

import warnings

warnings.filterwarnings("ignore")

try:
    import arch  # noqa: F401
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    print("arch not installed. Install with: pip install arch")

from vrp_strat.vol_engine.black_scholes import bs_price, implied_vol
from vrp_strat.vol_engine.combine import combine_forecasts
from vrp_strat.vol_engine.data import load_spy_ohlcv, load_vix
from vrp_strat.vol_engine.garch import garch_vol_forecast
from vrp_strat.vol_engine.har import har_rv_forecast
from vrp_strat.vol_engine.realized import close_to_close_vol, ewma_vol, yang_zhang_vol

__all__ = [
    "load_spy_ohlcv",
    "load_vix",
    "yang_zhang_vol",
    "close_to_close_vol",
    "ewma_vol",
    "garch_vol_forecast",
    "har_rv_forecast",
    "combine_forecasts",
    "bs_price",
    "implied_vol",
]
