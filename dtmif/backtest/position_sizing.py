"""Dynamic position sizing from spread and realized vol."""

import pandas as pd

from dtmif.backtest.config import MAX_POSITION


def position_size(
    vrp_spread: pd.Series,
    realized_vol: pd.Series,
    max_size: float = MAX_POSITION,
    target_vol: float = 0.15,
) -> pd.Series:
    """
    Dynamic position size based on signal strength and inverse-vol scaling.

    size_t = clip(|spread_t| / avg_spread, 0, max_size) × (target_vol / RV_t)
    """
    spread_norm = vrp_spread.abs() / vrp_spread.abs().rolling(60).mean().fillna(
        vrp_spread.abs().mean()
    )
    spread_norm = spread_norm.clip(upper=max_size)

    inv_vol = (target_vol / realized_vol.clip(lower=0.05)).clip(upper=max_size)

    size = (spread_norm * inv_vol).clip(lower=0, upper=max_size)
    size.name = "position_size"
    return size
