"""Dynamic position sizing from spread and realized vol."""

import pandas as pd

from dtmif.backtest.config import MAX_POSITION


def position_size(
    vrp_spread: pd.Series,
    realized_vol: pd.Series,
    max_size: float = MAX_POSITION,
    target_vol: float = 0.15,
) -> pd.Series:
    """Size from normalized spread and inverse vol, capped at ``max_size``."""
    spread_norm = vrp_spread.abs() / vrp_spread.abs().rolling(60).mean().fillna(
        vrp_spread.abs().mean()
    )
    spread_norm = spread_norm.clip(upper=max_size)

    inv_vol = (target_vol / realized_vol.clip(lower=0.05)).clip(upper=max_size)

    size = (spread_norm * inv_vol).clip(lower=0, upper=max_size)
    size.name = "position_size"
    return size
