"""Ensemble combination of volatility forecasts."""

import pandas as pd


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
        w = {k: 1 / len(forecasts) for k in forecasts}
    else:
        total = sum(weights.values())
        w = {k: v / total for k, v in weights.items()}

    combined = sum(df[k] * w[k] for k in forecasts)
    combined.name = "ensemble_vol"
    return combined
