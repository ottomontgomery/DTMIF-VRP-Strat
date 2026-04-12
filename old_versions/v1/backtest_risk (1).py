"""
backtest_risk.py
=================
Volatility Strategy: Long/Short Straddle based on IV vs. EWMA Spread
  - Signal generation with configurable threshold
  - Daily P&L computation (vega-approximation)
  - Risk metrics: Sharpe, VaR (95%/99%), CVaR, Max Drawdown

Author  : Quant Finance Project
Requires: numpy, pandas, matplotlib, volatility_engine.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for headless runs)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# Local module
from volatility_engine import (
    load_spy_prices,
    compute_log_returns,
    rolling_historical_vol,
    ewma_volatility,
    black_scholes_price,
)

# ============================================================
# STRATEGY PARAMETERS  (edit as needed)
# ============================================================

THRESHOLD      = 0.02    # |IV - EWMA| spread required to enter a trade
LAMBDA_EWMA    = 0.94    # RiskMetrics decay factor
RISK_FREE_RATE = 0.05    # annualised flat risk-free rate
START_DATE     = "2019-01-01"

# Vol-premium simulation (used when real IV history is unavailable)
# Mimics the empirical finding: IV ≈ realized vol + 3 vol-points on average
SIM_VOL_PREMIUM = 0.03   # average premium above EWMA
SIM_VOL_NOISE   = 0.005  # day-to-day noise around the premium


# ============================================================
# 1. SIGNAL GENERATION
# ============================================================

def generate_signals(
    vol_spread: pd.Series,
    threshold: float = THRESHOLD,
) -> pd.Series:
    """
    Generate daily straddle direction signals from the vol spread.

    Rule
    ----
    vol_spread = IV − EWMA_vol
    
      vol_spread > +threshold  →  -1  (short straddle: sell expensive implied vol)
      vol_spread < -threshold  →  +1  (long  straddle: buy cheap implied vol)
      |vol_spread| ≤ threshold →   0  (flat: no position)

    Economic rationale
    ------------------
    The volatility risk premium literature (Bakshi & Kapadia 2003; Carr & Wu 2009)
    documents that IV persistently exceeds subsequent realized vol on equity indices.
    The dominant signal is therefore -1 (short vol), with the threshold acting as a
    filter against noise and transaction costs.

    Parameters
    ----------
    vol_spread : pd.Series  - IV minus forecast vol (annualised)
    threshold  : float      - Minimum spread to trigger a trade

    Returns
    -------
    pd.Series of signals: {-1, 0, +1}
    """
    signals = pd.Series(0, index=vol_spread.index, name="signal", dtype=int)
    signals[vol_spread >  threshold] = -1   # short straddle
    signals[vol_spread < -threshold] =  1   # long  straddle
    return signals


# ============================================================
# 2. P&L APPROXIMATION
# ============================================================

def compute_pnl(
    signals: pd.Series,
    realized_vol: pd.Series,
    implied_vol: pd.Series,
    spot: pd.Series,
    T_remaining: float = 21 / 252,  # ~1 month in years
) -> pd.Series:
    """
    Approximate daily P&L for the vol strategy.

    Methodology
    -----------
    For a delta-hedged ATM straddle, the dominant P&L driver at short horizons is
    the difference between the vol sold (implied) and the vol realised:

        P&L_t ≈ −signal_{t-1} × vega_ATM × (realised_vol_t − implied_vol_{t-1})

    where vega_ATM ≈ S × sqrt(T / (2π))  [Black-Scholes ATM vega per unit notional].

    We scale by 1/100 so that P&L is expressed in dollars per $100 notional.

    Parameters
    ----------
    signals      : pd.Series  - Signal series (values -1, 0, +1)
    realized_vol : pd.Series  - Realised vol series (annualised, e.g. HV-20)
    implied_vol  : pd.Series  - Implied vol series (annualised)
    spot         : pd.Series  - Underlying price series
    T_remaining  : float      - Average time to expiry (years) for vega calc

    Returns
    -------
    pd.Series named 'pnl'
    """
    # Lag signals: use yesterday's signal for today's P&L
    sig_lagged = signals.shift(1)

    # Vol surprise: how much realised vol deviated from what was sold/bought
    vol_surprise = realized_vol - implied_vol

    # ATM straddle vega approximation: 2 × BS_vega of ATM call = S√(T/2π)
    # (factor 2 for straddle, divided by 2 accounts for the full vega)
    vega_atm = spot * np.sqrt(T_remaining / (2 * np.pi))

    # P&L: short vol profits when realised < implied (vol_surprise < 0)
    pnl = -sig_lagged * vega_atm * vol_surprise / 100.0
    pnl.name = "pnl"
    return pnl.dropna()


# ============================================================
# 3. RISK METRICS
# ============================================================

def annualised_sharpe(
    pnl: pd.Series,
    risk_free_daily: float = 0.0,
    trading_days: int = 252,
) -> float:
    """
    Annualised Sharpe Ratio = (mean excess return / std) × sqrt(252).

    Parameters
    ----------
    pnl              : pd.Series  - Daily P&L
    risk_free_daily  : float      - Daily risk-free return (default 0)
    trading_days     : int        - Annualisation factor

    Returns
    -------
    float: annualised Sharpe ratio (NaN if std = 0)
    """
    excess = pnl - risk_free_daily
    std = excess.std()
    if std == 0 or np.isnan(std):
        return np.nan
    return (excess.mean() / std) * np.sqrt(trading_days)


def historical_var_cvar(
    pnl: pd.Series,
    confidence_levels: list = [0.95, 0.99],
) -> dict:
    """
    Historical (non-parametric) Value-at-Risk and Conditional VaR.

    VaR_α  = −percentile(P&L, 1−α)       [loss convention: positive = loss]
    CVaR_α = −mean(P&L | P&L ≤ −VaR_α)   [Expected Shortfall]

    Parameters
    ----------
    pnl               : pd.Series  - Daily P&L
    confidence_levels : list       - e.g. [0.95, 0.99]

    Returns
    -------
    dict with keys like 'VaR_95', 'CVaR_95', 'VaR_99', 'CVaR_99'
    """
    results = {}
    clean = pnl.dropna()
    for cl in confidence_levels:
        tag = int(cl * 100)
        # percentile in the loss tail
        var_val  = -np.percentile(clean, (1 - cl) * 100)
        tail     = clean[clean <= -var_val]
        cvar_val = -tail.mean() if len(tail) > 0 else np.nan
        results[f"VaR_{tag}"]  = var_val
        results[f"CVaR_{tag}"] = cvar_val
    return results


def maximum_drawdown(pnl: pd.Series) -> dict:
    """
    Maximum absolute and percentage drawdown from cumulative P&L curve.

    Parameters
    ----------
    pnl : pd.Series  - Daily P&L

    Returns
    -------
    dict:
        max_abs_drawdown  : float     - Worst drawdown in dollars
        max_pct_drawdown  : float     - Worst drawdown as fraction of peak
        drawdown_series   : pd.Series - Full drawdown timeseries
        cum_pnl           : pd.Series - Cumulative P&L
    """
    cum   = pnl.cumsum()
    peak  = cum.cummax()
    dd    = cum - peak  # always ≤ 0

    max_abs_dd = dd.min()

    # Percentage drawdown relative to rolling peak (avoid division by zero)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_dd = dd / peak.replace(0, np.nan)

    max_pct_dd = pct_dd.min()

    return {
        "max_abs_drawdown": max_abs_dd,
        "max_pct_drawdown": max_pct_dd,
        "drawdown_series":  dd,
        "cum_pnl":          cum,
    }


def print_risk_report(pnl: pd.Series, label: str = "Strategy") -> pd.DataFrame:
    """
    Print a formatted risk report and return it as a DataFrame.

    Parameters
    ----------
    pnl   : pd.Series  - Daily P&L
    label : str        - Report title

    Returns
    -------
    pd.DataFrame with columns ['Metric', 'Value']
    """
    sr  = annualised_sharpe(pnl)
    vc  = historical_var_cvar(pnl)
    dd  = maximum_drawdown(pnl)

    rows = [
        ("Annualised Sharpe Ratio",  f"{sr:.4f}"),
        ("Total P&L",               f"${pnl.sum():.2f}"),
        ("Daily Mean P&L",          f"${pnl.mean():.4f}"),
        ("Daily Std Dev",           f"${pnl.std():.4f}"),
        ("VaR 95% (daily, $ loss)", f"${vc['VaR_95']:.4f}"),
        ("CVaR 95% (daily, $ loss)",f"${vc['CVaR_95']:.4f}"),
        ("VaR 99% (daily, $ loss)", f"${vc['VaR_99']:.4f}"),
        ("CVaR 99% (daily, $ loss)",f"${vc['CVaR_99']:.4f}"),
        ("Max Abs Drawdown",        f"${dd['max_abs_drawdown']:.4f}"),
        ("Max % Drawdown",          f"{dd['max_pct_drawdown']:.2%}"),
        ("Obs (trading days)",      str(len(pnl))),
    ]

    width = 50
    print(f"\n{'=' * width}")
    print(f"  RISK REPORT: {label}")
    print(f"{'=' * width}")
    for k, v in rows:
        print(f"  {k:<35} {v}")
    print(f"{'=' * width}\n")

    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ============================================================
# 4. PLOTTING
# ============================================================

def plot_backtest(df: pd.DataFrame, vol_spread: pd.Series, pnl: pd.Series,
                  out_path: str = "backtest_plot.png") -> None:
    """
    Generate a three-panel backtest plot:
      1. Volatility comparison (HV-20, EWMA, IV)
      2. Volatility spread with threshold bands
      3. Cumulative P&L with drawdown overlay
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)
    fig.patch.set_facecolor("#F7F6F2")
    for ax in axes:
        ax.set_facecolor("#F9F8F5")
        ax.spines[["top", "right"]].set_visible(False)

    TEAL  = "#20808D"
    RUST  = "#A84B2F"
    DARK  = "#1B474D"
    GOLD  = "#FFC553"
    RED   = "#A13544"

    # Panel 1: Volatility comparison
    ax = axes[0]
    df["hv_20d"].plot(ax=ax, color=DARK,  lw=1.4, label="HV-20 (realised)")
    df["ewma_vol"].plot(ax=ax, color=TEAL, lw=1.6, label="EWMA (λ=0.94)")
    df["implied_vol"].plot(ax=ax, color=RUST, lw=1.4, ls="--", label="Implied Vol")
    ax.set_title("Volatility Comparison — SPY", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Annualised Vol")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    # Panel 2: Volatility spread
    ax = axes[1]
    vol_spread.plot(ax=ax, color=GOLD, lw=1.5, label="IV − EWMA spread")
    ax.axhline( THRESHOLD, color=RED,  ls="--", lw=1.2, alpha=0.7, label=f"+Threshold ({THRESHOLD:.0%})")
    ax.axhline(-THRESHOLD, color=TEAL, ls="--", lw=1.2, alpha=0.7, label=f"−Threshold (−{THRESHOLD:.0%})")
    ax.axhline(0, color="grey", lw=0.8, alpha=0.5)
    ax.fill_between(vol_spread.index, THRESHOLD, vol_spread.where(vol_spread > THRESHOLD),
                    alpha=0.15, color=RED)
    ax.fill_between(vol_spread.index, -THRESHOLD, vol_spread.where(vol_spread < -THRESHOLD),
                    alpha=0.15, color=TEAL)
    ax.set_title("Volatility Spread (IV − EWMA) with Trade Zones", fontsize=13,
                 fontweight="bold", pad=10)
    ax.set_ylabel("Spread (annualised)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    # Panel 3: Cumulative P&L + drawdown
    ax = axes[2]
    dd_data = maximum_drawdown(pnl)
    cum_pnl = dd_data["cum_pnl"]
    dd_ser  = dd_data["drawdown_series"]

    cum_pnl.plot(ax=ax, color=TEAL, lw=1.8, label="Cumulative P&L")
    ax.fill_between(dd_ser.index, dd_ser, 0, alpha=0.25, color=RED, label="Drawdown")
    ax.axhline(0, color="grey", lw=0.8, alpha=0.5)
    ax.set_title("Cumulative P&L and Drawdown", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("P&L ($ per $100 notional)")
    ax.legend(fontsize=9, framealpha=0.6)

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


# ============================================================
# MAIN — BACKTEST RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  VOLATILITY STRATEGY BACKTEST — SPY")
    print("=" * 60)

    # ---- Load data and compute vol measures ----
    prices  = load_spy_prices(start=START_DATE)
    log_ret = compute_log_returns(prices["spy_close"])
    hv_20   = rolling_historical_vol(log_ret, window=20)
    hv_60   = rolling_historical_vol(log_ret, window=60)
    ewma_v  = ewma_volatility(log_ret, lam=LAMBDA_EWMA)

    # ---- Simulated IV ----
    # In production, replace with real historical IV from CBOE or a data vendor.
    # Here we simulate IV = EWMA + constant premium + white noise,
    # consistent with the empirical volatility risk premium literature.
    np.random.seed(42)
    noise        = pd.Series(
        np.random.normal(0, SIM_VOL_NOISE, len(ewma_v)),
        index=ewma_v.index
    )
    simulated_iv = (ewma_v + SIM_VOL_PREMIUM + noise).rename("implied_vol")

    # ---- Combine into master DataFrame ----
    df = pd.DataFrame({
        "spy_close":  prices["spy_close"],
        "log_return": log_ret,
        "hv_20d":     hv_20,
        "hv_60d":     hv_60,
        "ewma_vol":   ewma_v,
        "implied_vol": simulated_iv,
    }).dropna()

    # ---- Signal generation ----
    vol_spread   = df["implied_vol"] - df["ewma_vol"]
    df["signal"] = generate_signals(vol_spread, threshold=THRESHOLD)

    signal_counts = df["signal"].value_counts()
    print(f"\nSignal distribution:  Short={signal_counts.get(-1, 0)}  "
          f"Flat={signal_counts.get(0, 0)}  Long={signal_counts.get(1, 0)}")

    # ---- P&L computation ----
    df["pnl"] = compute_pnl(
        signals=df["signal"],
        realized_vol=df["hv_20d"],
        implied_vol=df["implied_vol"],
        spot=df["spy_close"],
    )

    pnl = df["pnl"].dropna()

    # ---- Risk report ----
    report_df = print_risk_report(pnl, label="IV vs. EWMA Straddle (SPY)")
    report_df.to_csv("risk_report.csv", index=False)
    print("Risk report saved → risk_report.csv")

    # ---- Save backtest data ----
    df.to_csv("backtest_results.csv")
    print("Full results saved  → backtest_results.csv")

    # ---- Plot ----
    plot_backtest(df, vol_spread, pnl, out_path="backtest_plot.png")

    print("\nAll outputs saved. Run with real IV data for production use.")
