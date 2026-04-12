"""GJR-GARCH conditional volatility (arch package)."""

import numpy as np
import pandas as pd

try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False

from dtmif.vol_engine.realized import ewma_vol


def garch_vol_forecast(
    close: pd.Series,
    p: int = 1,
    q: int = 1,
    annualize: bool = True,
    trading_days: int = 252,
    use_gjr: bool = True,
) -> pd.Series:
    """One-step-ahead GARCH or GJR-GARCH conditional vol; EWMA if arch missing or fit fails."""
    if not ARCH_OK:
        print("WARNING: arch not available. Returning EWMA as fallback.")
        return ewma_vol(close, annualize=annualize)

    log_ret = np.log(close / close.shift(1)).dropna() * 100

    try:
        if use_gjr:
            model = arch_model(log_ret, vol="Garch", p=p, o=1, q=q, dist="normal")
        else:
            model = arch_model(log_ret, vol="Garch", p=p, q=q, dist="normal")
        res = model.fit(disp="off", show_warning=False)
    except Exception as e:
        print(f"GARCH fit failed ({e}). Using EWMA fallback.")
        return ewma_vol(close, annualize=annualize)

    cond_vol = res.conditional_volatility / 100.0
    if annualize:
        cond_vol = cond_vol * np.sqrt(trading_days)

    cond_vol.name = "gjr_garch_vol"
    return cond_vol
