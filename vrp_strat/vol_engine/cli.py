"""Command-line demo for the volatility engine."""

import numpy as np
import pandas as pd

from vrp_strat.vol_engine import (
    close_to_close_vol,
    combine_forecasts,
    ewma_vol,
    garch_vol_forecast,
    har_rv_forecast,
    load_spy_ohlcv,
    load_vix,
    yang_zhang_vol,
)


def main() -> None:
    print("=" * 60)
    print("  VOLATILITY ENGINE v2 — DEMO")
    print("=" * 60)

    ohlcv = load_spy_ohlcv(start="2019-01-01")

    try:
        vix = load_vix(start="2019-01-01")
    except Exception as e:
        print(f"VIX load failed: {e}")
        vix = None

    print("\nComputing vol estimators...")
    hv20 = close_to_close_vol(ohlcv["close"], window=20)
    yz20 = yang_zhang_vol(ohlcv, window=20)
    ewma = ewma_vol(ohlcv["close"])

    print("Fitting GARCH(1,1) / GJR-GARCH...")
    garch = garch_vol_forecast(ohlcv["close"], use_gjr=True)

    print("Fitting HAR-RV (rolling OLS)...")
    har = har_rv_forecast(ohlcv["close"])

    forecasts = {k: v for k, v in {"ewma": ewma, "garch": garch, "har": har}.items() if v is not None}
    ensemble = combine_forecasts(forecasts)

    result = pd.DataFrame(
        {
            "HV-20": hv20,
            "YZ-20": yz20,
            "EWMA": ewma,
            "GJR-GARCH": garch,
            "HAR-RV": har,
            "Ensemble": ensemble,
        }
    ).dropna()

    if vix is not None:
        vix_aligned = vix.reindex(result.index, method="ffill")
        result["VIX"] = vix_aligned

    print(f"\nFinal DataFrame: {len(result)} rows")
    print("\nLast 5 rows:")
    print(result.tail(5).round(4))

    result.to_csv("spy_volatility_v2.csv")
    print("\nSaved → spy_volatility_v2.csv")

    from sklearn.metrics import mean_squared_error

    common = result.dropna()
    rv_proxy = common["HV-20"]
    print("\n=== FORECAST RMSE vs HV-20 REALIZED ===")
    for col in ["EWMA", "GJR-GARCH", "HAR-RV", "Ensemble"]:
        if col in common.columns:
            rmse = np.sqrt(mean_squared_error(rv_proxy, common[col]))
            print(f"  {col:<15}: RMSE = {rmse:.6f}")


if __name__ == "__main__":
    main()
