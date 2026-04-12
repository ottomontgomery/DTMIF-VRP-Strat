"""VRP signal generation with vol-of-vol regime filter."""

import pandas as pd

from dtmif.backtest.config import THRESHOLD, VOL_OF_VOL_LIMIT


def generate_signals_v2(
    vrp_spread: pd.Series,
    vol_of_vol: pd.Series,
    threshold: float = THRESHOLD,
    vov_limit: float = VOL_OF_VOL_LIMIT,
) -> pd.Series:
    """VRP spread vs threshold, zeroed when vol-of-vol exceeds ``vov_limit``."""
    raw = pd.Series(0, index=vrp_spread.index, dtype=int)
    raw[vrp_spread > threshold] = -1
    raw[vrp_spread < -threshold] = 1

    vov_aligned = vol_of_vol.reindex(vrp_spread.index, method="ffill")
    raw[vov_aligned > vov_limit] = 0

    raw.name = "signal"
    return raw
