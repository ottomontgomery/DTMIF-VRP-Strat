"""Ensemble combination of volatility forecasts."""

import pandas as pd


def combine_forecasts(
    forecasts: dict,
    weights: dict = None,
) -> pd.Series:
    """Weighted average of forecast series; equal weights if ``weights`` is omitted."""
    df = pd.DataFrame(forecasts).dropna()
    if weights is None:
        w = {k: 1 / len(forecasts) for k in forecasts}
    else:
        total = sum(weights.values())
        w = {k: v / total for k, v in weights.items()}

    combined = sum(df[k] * w[k] for k in forecasts)
    combined.name = "ensemble_vol"
    return combined
