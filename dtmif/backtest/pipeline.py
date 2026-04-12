"""End-to-end volatility strategy backtest (v2)."""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dtmif.vol_engine import (
    close_to_close_vol,
    combine_forecasts,
    ewma_vol,
    garch_vol_forecast,
    har_rv_forecast,
    load_spy_ohlcv,
    yang_zhang_vol,
)

from dtmif.backtest.config import IS_END, THRESHOLD, VOL_OF_VOL_LIMIT
from dtmif.backtest.pnl import compute_pnl_v2
from dtmif.backtest.plotting import plot_results_v2
from dtmif.backtest.position_sizing import position_size
from dtmif.backtest.risk import compute_risk_metrics, print_risk_report
from dtmif.backtest.signals import generate_signals_v2


def run_backtest() -> None:
    """Load data, build signals, P&L, metrics; write CSV/PNG to the current working directory."""
    print("=" * 60)
    print("  VOLATILITY STRATEGY BACKTEST v2 — SPY")
    print("=" * 60)

    print("\n[1] Loading data...")
    ohlcv = load_spy_ohlcv(start="2019-01-01")

    try:
        import yfinance as yf

        vix_raw = yf.download("^VIX", start="2019-01-01", progress=False)
        vix = vix_raw["Close"].squeeze() / 100.0
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        vix.name = "vix"
        print(f"Loaded VIX: {len(vix)} rows")
    except Exception as e:
        print(f"VIX load failed ({e}). Simulating VRP proxy.")
        vix = None

    print("\n[2] Computing volatility forecasts...")
    hv20 = close_to_close_vol(ohlcv["close"], window=20)
    yz20 = yang_zhang_vol(ohlcv, window=20)
    ewma = ewma_vol(ohlcv["close"])

    print("  Fitting GJR-GARCH(1,1,1)...")
    garch = garch_vol_forecast(ohlcv["close"], use_gjr=True)

    print("  Fitting HAR-RV (rolling OLS)...")
    har = har_rv_forecast(ohlcv["close"])

    ensemble = combine_forecasts({"ewma": ewma, "garch": garch, "har": har})

    print("\n[3] Building master DataFrame...")
    df = pd.DataFrame(
        {
            "spy_close": ohlcv["close"],
            "hv_20d": hv20,
            "yz_vol": yz20,
            "ewma_vol": ewma,
            "garch_vol": garch,
            "har_vol": har,
            "vol_forecast": ensemble,
        }
    ).dropna()

    if vix is not None:
        df["vix"] = vix.reindex(df.index, method="ffill")
        df["iv_proxy"] = df["vix"]
    else:
        np.random.seed(42)
        n = len(df)
        vrp_sim = np.zeros(n)
        vrp_sim[0] = 0.04
        for t in range(1, n):
            vrp_sim[t] = 0.95 * vrp_sim[t - 1] + 0.04 * (1 - 0.95) + np.random.normal(0, 0.008)
        df["iv_proxy"] = df["vol_forecast"] + pd.Series(vrp_sim, index=df.index)

    df = df.dropna()

    df["vrp_spread"] = df["iv_proxy"] - df["vol_forecast"]
    df["vol_of_vol"] = df["vol_forecast"].rolling(20).std()

    print("\n[4] Generating signals (with regime filter)...")
    df["signal"] = generate_signals_v2(
        vrp_spread=df["vrp_spread"],
        vol_of_vol=df["vol_of_vol"],
        threshold=THRESHOLD,
        vov_limit=VOL_OF_VOL_LIMIT,
    )

    signal_counts = df["signal"].value_counts().sort_index()
    print(
        f"  Signal distribution:  Short={signal_counts.get(-1, 0)}  "
        f"Flat={signal_counts.get(0, 0)}  Long={signal_counts.get(1, 0)}"
    )
    pct_active = (df["signal"] != 0).mean()
    print(f"  % time active: {pct_active:.1%}  (was 98.2% in v1 — now regime-filtered)")

    print("\n[5] Computing dynamic position sizes...")
    df["pos_size"] = position_size(
        vrp_spread=df["vrp_spread"],
        realized_vol=df["vol_forecast"],
    ) * df["signal"].abs()

    print("\n[6] Computing P&L (theta-gamma decomposition)...")
    pnl_df = compute_pnl_v2(
        signals=df["signal"],
        sizes=df["pos_size"],
        realized_vol=df["vol_forecast"],
        implied_vol=df["iv_proxy"],
        spot=df["spy_close"],
        use_theta_gamma=True,
    )
    df = df.join(pnl_df[["pnl", "theta_pnl", "gamma_pnl"]], how="left")

    print("\n[7] Computing risk metrics...")
    split = pd.Timestamp(IS_END)
    pnl_all = df["pnl"].dropna()
    pnl_is = pnl_all[pnl_all.index <= split]
    pnl_oos = pnl_all[pnl_all.index > split]

    metrics_all = compute_risk_metrics(pnl_all, label="Full Sample")
    metrics_is = compute_risk_metrics(pnl_is, label="In-Sample (2019–2023)")
    metrics_oos = compute_risk_metrics(pnl_oos, label="Out-of-Sample (2024+)")

    print_risk_report(metrics_all)
    print_risk_report(metrics_is)
    print_risk_report(metrics_oos)

    print("\n[8] Saving outputs...")
    df.to_csv("backtest_v2_results.csv")
    print("  Full results → backtest_v2_results.csv")

    report_rows = []
    for m in [metrics_all, metrics_is, metrics_oos]:
        report_rows.append(
            {
                "Sample": m["label"],
                "N Obs": m["n_obs"],
                "Ann Return": f"{m['ann_return']:.2%}",
                "Ann Vol": f"{m['ann_vol']:.2%}",
                "Sharpe": f"{m['sharpe']:.4f}",
                "Sortino": f"{m['sortino']:.4f}",
                "Calmar": f"{m['calmar']:.4f}",
                "VaR 95%": f"${m['var_95']:.6f}",
                "CVaR 95%": f"${m['cvar_95']:.6f}",
                "VaR 99%": f"${m['var_99']:.6f}",
                "CVaR 99%": f"${m['cvar_99']:.6f}",
                "Max Abs DD": f"${m['max_abs_dd']:.4f}",
                "Win Rate": f"{m['win_rate']:.1%}",
                "Profit Factor": f"{m['profit_factor']:.4f}",
            }
        )
    pd.DataFrame(report_rows).to_csv("backtest_v2_risk_report.csv", index=False)
    print("  Risk report → backtest_v2_risk_report.csv")

    print("\n[9] Generating chart...")
    plot_results_v2(df, metrics_is, metrics_oos, out_path="backtest_v2_plot.png")

    print("\n[DONE] All outputs saved.")
    print("\nKey improvement vs v1:")
    print("  v1 Sharpe: -16.27  (sign bug + circular IV)")
    print(f"  v2 Sharpe: {metrics_all['sharpe']:.2f}  (fixed)")
    print(f"  % time active reduced from 98.2% → {pct_active:.1%}")
