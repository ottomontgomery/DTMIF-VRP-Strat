"""
backtest_v2.py
==============
Improved Volatility Strategy Backtest — v2

Addresses all five root causes identified in the post-mortem:
  1. FIXED: P&L sign inversion bug
  2. FIXED: Real VIX data replaces circular simulated IV
  3. IMPROVED: GJR-GARCH vol forecast replaces EWMA-only
  4. IMPROVED: Adaptive vol-of-vol regime filter (stop trading in crash regimes)
  5. IMPROVED: Dynamic position sizing (inverse-vol Kelly fraction)
  6. IMPROVED: Theta-gamma P&L decomposition (more realistic than pure vega approx)
  7. FIXED: Drawdown percentage calculation
  8. NEW: Walk-forward out-of-sample validation split

Author  : Quant Finance Project
Requires: numpy, pandas, scipy, arch, yfinance, sklearn, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from vol_engine_v2 import (
    load_spy_ohlcv, load_vix, yang_zhang_vol, close_to_close_vol,
    ewma_vol, garch_vol_forecast, har_rv_forecast, combine_forecasts,
)

# ============================================================
# PARAMETERS
# ============================================================

RISK_FREE_RATE   = 0.05     # annualised flat
THRESHOLD        = 0.02     # base spread threshold (2 vol points)
LAMBDA_EWMA      = 0.94
MAX_POSITION     = 1.0      # max position size scalar
VOL_OF_VOL_LIMIT = 0.035    # regime filter: don't trade when vol-of-vol > 3.5%
VRP_MIN_SIGNAL   = 0.01     # minimum spread to count as signal (reduce noise)
IS_END           = "2023-12-31"  # in-sample / out-of-sample split date
T_OPTION         = 21 / 252      # option time to expiry for vega calc (1 month)


# ============================================================
# 1. SIGNAL GENERATION (improved)
# ============================================================

def generate_signals_v2(
    vrp_spread: pd.Series,
    vol_of_vol: pd.Series,
    threshold: float = THRESHOLD,
    vov_limit: float = VOL_OF_VOL_LIMIT,
) -> pd.Series:
    """
    Improved signal: add regime filter on top of spread threshold.

    Signal logic:
      Step 1 — Compute raw signal from spread
        spread > +threshold  →  raw = -1  (short straddle, IV overpriced)
        spread < -threshold  →  raw = +1  (long  straddle, IV underpriced)
        |spread| <= threshold → raw =  0  (flat)

      Step 2 — Regime filter (vol-of-vol)
        If vol_of_vol > vov_limit (tail-risk regime), force signal = 0.
        Selling vol during crash regimes (2008, 2020, 2022) is the primary
        source of catastrophic drawdowns in short-vol strategies.

    Parameters
    ----------
    vrp_spread  : pd.Series  - VIX - realized vol forecast (annualised)
    vol_of_vol  : pd.Series  - 20-day rolling std of the realized vol forecast
    threshold   : float      - Minimum spread to trade (vol points)
    vov_limit   : float      - Max allowed vol-of-vol before going flat

    Returns
    -------
    pd.Series of signals: {-1, 0, +1}
    """
    raw = pd.Series(0, index=vrp_spread.index, dtype=int)
    raw[vrp_spread >  threshold] = -1
    raw[vrp_spread < -threshold] =  1

    # Regime filter: go flat when vol-of-vol is elevated
    vov_aligned = vol_of_vol.reindex(vrp_spread.index, method="ffill")
    raw[vov_aligned > vov_limit] = 0

    raw.name = "signal"
    return raw


# ============================================================
# 2. DYNAMIC POSITION SIZING
# ============================================================

def position_size(
    vrp_spread: pd.Series,
    realized_vol: pd.Series,
    max_size: float = MAX_POSITION,
    target_vol: float = 0.15,   # target annualised portfolio vol
) -> pd.Series:
    """
    Dynamic position size based on:
      (a) Signal strength (larger spread → larger size, proportional scaling)
      (b) Inverse-vol sizing (scale down when realized vol is high)

    The inverse-vol scaling keeps the dollar-vega exposure approximately
    constant regardless of the vol regime, reducing the tendency to
    over-expose during low-vol periods and blow up in high-vol regimes.

    size_t = clip(|spread_t| / avg_spread, 0, max_size) × (target_vol / RV_t)

    Returns a scalar in [0, max_size].
    """
    # Signal strength scaling (normalize spread by rolling mean spread)
    spread_norm = vrp_spread.abs() / vrp_spread.abs().rolling(60).mean().fillna(vrp_spread.abs().mean())
    spread_norm = spread_norm.clip(upper=max_size)

    # Inverse-vol scaling
    inv_vol = (target_vol / realized_vol.clip(lower=0.05)).clip(upper=max_size)

    size = (spread_norm * inv_vol).clip(lower=0, upper=max_size)
    size.name = "position_size"
    return size


# ============================================================
# 3. P&L COMPUTATION (FIXED + IMPROVED)
# ============================================================

def compute_pnl_v2(
    signals: pd.Series,
    sizes: pd.Series,
    realized_vol: pd.Series,
    implied_vol: pd.Series,
    spot: pd.Series,
    r: float = RISK_FREE_RATE,
    T: float = T_OPTION,
    use_theta_gamma: bool = True,
) -> pd.DataFrame:
    """
    Improved P&L computation with:
      (a) FIXED sign convention
      (b) Theta-gamma decomposition (more realistic than pure vega approximation)
      (c) Dynamic position sizing

    Theta-gamma decomposition for a delta-hedged position:

      P&L_daily = Theta_$ × (1/252) + Gamma_$ × (daily_return)²

    For short straddle (signal = -1, scaled by size):

      Theta_$ = S²×Gamma × σ²_IV / 2    [time decay earned = positive for short]
      Gamma_$ = -S²×Gamma               [negative exposure to squared return]

      Delta-hedged short straddle daily P&L:
        = size × Gamma_ATM × S² × (σ²_IV/252 - r_t²)

      where Gamma_ATM = N'(d1) / (S × σ_IV × √T)
            r_t = daily log return at time t

    This is the fundamental volatility P&L trade-off: you earn theta daily
    but pay gamma on every realized move. Net profit iff σ_IV > σ_realized.

    Parameters
    ----------
    signals      : pd.Series  - {-1, 0, +1}
    sizes        : pd.Series  - Position size scalar in [0, max_size]
    realized_vol : pd.Series  - Annualised realized vol (forecast, not actual)
    implied_vol  : pd.Series  - VIX or options-implied vol (annualised)
    spot         : pd.Series  - Underlying price series
    r            : float      - Risk-free rate
    T            : float      - Option maturity in years
    use_theta_gamma: bool     - Use theta-gamma if True, else vega approx

    Returns
    -------
    pd.DataFrame with columns: pnl, theta_pnl, gamma_pnl, position, size
    """
    # Lag signal and size (use yesterday's decision for today's P&L)
    sig_lag  = signals.shift(1)
    size_lag = sizes.shift(1).fillna(0)

    # Daily log returns
    log_ret = np.log(spot / spot.shift(1))

    if use_theta_gamma:
        # --- Theta-gamma decomposition ---
        # Daily variance budget: IV_daily² = (IV_annual)² / 252
        iv_daily_var   = (implied_vol**2) / 252.0

        # Realized daily variance on this date
        rv_daily_var   = log_ret**2

        # ATM Gamma approximation: N'(d1) / (S × σ × √T)
        # At d1 ≈ 0 (ATM), N'(0) = 1/√(2π) ≈ 0.3989
        # Dollar Gamma: S² × Gamma = S / (σ × √T) × N'(d1) ≈ S / (σ × √T × √(2π))
        sigma_iv    = implied_vol.clip(lower=0.01)   # avoid division by zero
        dollar_gamma = spot / (sigma_iv * np.sqrt(T) * np.sqrt(2 * np.pi))  # per unit

        # Theta-gamma P&L per unit position:
        #   long straddle  (+1): pnl = dollar_gamma × (rv_daily_var - iv_daily_var)
        #   short straddle (-1): pnl = dollar_gamma × (iv_daily_var - rv_daily_var)
        #   = signal × (-1) × dollar_gamma × (rv_daily_var - iv_daily_var)
        #   = signal × dollar_gamma × (iv_daily_var - rv_daily_var)

        theta_term = dollar_gamma * iv_daily_var        # daily theta earned by short
        gamma_term = dollar_gamma * rv_daily_var        # daily gamma paid by short

        # For short (signal=-1), profit = theta earned - gamma paid
        pnl_unit = sig_lag * (rv_daily_var - iv_daily_var) * dollar_gamma

        # Scale by position size and notional convention (per $100 spot)
        pnl = pnl_unit * size_lag / 100.0

        result = pd.DataFrame({
            "pnl":       pnl,
            "theta_pnl": sig_lag * (-1) * theta_term * size_lag / 100.0,
            "gamma_pnl": sig_lag * (-1) * (-rv_daily_var) * dollar_gamma * size_lag / 100.0,
            "position":  sig_lag,
            "size":      size_lag,
        })

    else:
        # --- Vega approximation (v1, sign CORRECTED) ---
        # P&L ≈ signal × Vega_ATM × (RV - IV)
        # Note: CORRECTED sign — positive for short when IV > RV
        vol_surprise = realized_vol - implied_vol
        vega_atm = spot * np.sqrt(T / (2 * np.pi))
        pnl = sig_lag * size_lag * vega_atm * vol_surprise / 100.0

        result = pd.DataFrame({
            "pnl":       pnl,
            "theta_pnl": pd.Series(np.nan, index=pnl.index),
            "gamma_pnl": pd.Series(np.nan, index=pnl.index),
            "position":  sig_lag,
            "size":      size_lag,
        })

    return result.dropna(subset=["pnl"])


# ============================================================
# 4. RISK METRICS (FIXED)
# ============================================================

def compute_risk_metrics(pnl: pd.Series, label: str = "Strategy") -> dict:
    """
    Comprehensive risk metrics with fixed drawdown % calculation.
    Returns dict with both scalar values and series for plotting.
    """
    clean = pnl.dropna()
    n = len(clean)

    # Sharpe
    sharpe = clean.mean() / clean.std() * np.sqrt(252) if clean.std() > 0 else np.nan

    # VaR / CVaR — historical (correct loss sign convention: positive = loss)
    var95  = -np.percentile(clean, 5)
    var99  = -np.percentile(clean, 1)
    cvar95 = -clean[clean <= -var95].mean()
    cvar99 = -clean[clean <= -var99].mean()

    # Calmar ratio = Annualized Return / |Max DD|
    ann_return = clean.mean() * 252
    cum = clean.cumsum()
    peak = cum.cummax()
    dd   = cum - peak

    max_abs_dd = dd.min()

    # FIXED: % drawdown — need to handle the case where peak is always ≤ 0
    # Use peak relative to absolute starting point (0) instead
    cum0 = pd.concat([pd.Series([0.0]), cum])  # prepend 0
    peak0 = cum0.cummax()
    dd0   = cum0 - peak0
    # Only meaningful when peak0 > 0 (otherwise no invested capital at risk)
    peak0_pos = peak0[peak0 > 0]
    if len(peak0_pos) > 0:
        pct_dd = (dd0[peak0_pos.index] / peak0_pos).min()
    else:
        pct_dd = np.nan  # never in profit → % DD undefined

    calmar = ann_return / abs(max_abs_dd) if max_abs_dd != 0 else np.nan

    # Sortino
    downside = clean[clean < 0].std() * np.sqrt(252)
    sortino  = ann_return / downside if downside > 0 else np.nan

    # Win rate and profit factor
    wins    = clean[clean > 0]
    losses  = clean[clean < 0]
    win_rate   = len(wins) / n
    pf         = wins.sum() / losses.abs().sum() if len(losses) > 0 else np.inf

    metrics = {
        "label":          label,
        "n_obs":          n,
        "ann_return":     ann_return,
        "ann_vol":        clean.std() * np.sqrt(252),
        "sharpe":         sharpe,
        "sortino":        sortino,
        "calmar":         calmar,
        "var_95":         var95,
        "cvar_95":        cvar95,
        "var_99":         var99,
        "cvar_99":        cvar99,
        "max_abs_dd":     max_abs_dd,
        "max_pct_dd":     pct_dd,
        "win_rate":       win_rate,
        "profit_factor":  pf,
        "cum_pnl":        cum,
        "drawdown":       dd,
    }
    return metrics


def print_risk_report(m: dict) -> None:
    """Pretty-print the risk metrics dict."""
    w = 55
    print(f"\n{'='*w}")
    print(f"  RISK REPORT: {m['label']}")
    print(f"{'='*w}")
    rows = [
        ("Obs (trading days)",        f"{m['n_obs']}"),
        ("Annualised Return",         f"{m['ann_return']:.2%}"),
        ("Annualised Vol",            f"{m['ann_vol']:.2%}"),
        ("Sharpe Ratio (ann.)",       f"{m['sharpe']:.4f}"),
        ("Sortino Ratio (ann.)",      f"{m['sortino']:.4f}"),
        ("Calmar Ratio",              f"{m['calmar']:.4f}"),
        ("VaR 95% (daily $)",        f"${m['var_95']:.6f}"),
        ("CVaR 95% (daily $)",       f"${m['cvar_95']:.6f}"),
        ("VaR 99% (daily $)",        f"${m['var_99']:.6f}"),
        ("CVaR 99% (daily $)",       f"${m['cvar_99']:.6f}"),
        ("Max Abs Drawdown ($)",      f"${m['max_abs_dd']:.4f}"),
        ("Max % Drawdown",
             f"{m['max_pct_dd']:.2%}" if m['max_pct_dd'] is not np.nan else "N/A (never in profit)"),
        ("Win Rate",                  f"{m['win_rate']:.1%}"),
        ("Profit Factor",             f"{m['profit_factor']:.4f}"),
    ]
    for k, v in rows:
        print(f"  {k:<35} {v}")
    print(f"{'='*w}\n")


# ============================================================
# 5. PLOTTING
# ============================================================

def plot_results_v2(
    df: pd.DataFrame,
    metrics_is: dict,
    metrics_oos: dict,
    out_path: str = "backtest_v2_plot.png",
) -> None:
    """4-panel diagnostic plot for the improved backtest."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.patch.set_facecolor("#F7F6F2")
    for ax in axes:
        ax.set_facecolor("#F9F8F5")
        ax.spines[["top","right"]].set_visible(False)

    TEAL  = "#20808D"
    RUST  = "#A84B2F"
    DARK  = "#0A2540"
    GOLD  = "#E8B84B"
    RED   = "#A13544"
    GREY  = "#7A7974"

    # ── Panel 1: Vol comparison ──
    ax = axes[0]
    df["vix"].plot(ax=ax, color=RUST, lw=1.4, ls="--", label="VIX (IV proxy)")
    df["vol_forecast"].plot(ax=ax, color=TEAL, lw=1.6, label="Vol Forecast (GJR-GARCH)")
    df["hv_20d"].plot(ax=ax, color=DARK, lw=1.0, alpha=0.6, label="HV-20 (realized)")
    ax.set_title("Volatility Comparison — VIX vs. Forecast vs. Realized",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Annualised Vol")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    # ── Panel 2: VRP spread + regime filter ──
    ax = axes[1]
    df["vrp_spread"].plot(ax=ax, color=GOLD, lw=1.5, label="VRP Spread (VIX − Forecast)")
    ax.axhline(THRESHOLD,  color=RED,  ls="--", lw=1.2, alpha=0.7, label=f"+Threshold")
    ax.axhline(-THRESHOLD, color=TEAL, ls="--", lw=1.2, alpha=0.7, label=f"−Threshold")
    ax.axhline(0, color=GREY, lw=0.8, alpha=0.5)

    # Shade regime-filtered periods (vol-of-vol exceeded)
    blocked = df["vol_of_vol"] > VOL_OF_VOL_LIMIT
    if blocked.any():
        ax.fill_between(df.index, ax.get_ylim()[0] if ax.get_ylim()[0] < -0.1 else -0.15,
                        0.15, where=blocked, alpha=0.12, color=RED, label="Regime filter active")

    ax.set_title("VRP Spread with Threshold and Regime Filter", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Spread (IV − Forecast)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    # ── Panel 3: Cumulative P&L (IS vs OOS) ──
    ax = axes[2]
    split = pd.Timestamp(IS_END)
    cum_full = df["pnl"].dropna().cumsum()

    is_mask  = cum_full.index <= split
    oos_mask = cum_full.index >  split

    if is_mask.any():
        cum_full[is_mask].plot(ax=ax, color=TEAL, lw=2, label="In-Sample (train)")
    if oos_mask.any():
        cum_full[oos_mask].plot(ax=ax, color=RUST, lw=2, label="Out-of-Sample (test)")

    dd = df["pnl"].dropna().cumsum()
    dd_series = dd - dd.cummax()
    ax.fill_between(dd_series.index, dd_series, 0, alpha=0.18, color=RED, label="Drawdown")
    ax.axvline(split, color=GREY, ls=":", lw=1.2, label=f"IS/OOS split ({IS_END})")
    ax.axhline(0, color=GREY, lw=0.8, alpha=0.5)
    ax.set_title("Cumulative P&L — In-Sample vs. Out-of-Sample", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("P&L ($ per $100 notional)")
    ax.legend(fontsize=9, framealpha=0.6)

    # ── Panel 4: Theta vs Gamma decomposition ──
    ax = axes[3]
    if "theta_pnl" in df.columns and not df["theta_pnl"].isna().all():
        theta_cum = df["theta_pnl"].dropna().cumsum()
        gamma_cum = df["gamma_pnl"].dropna().cumsum()
        theta_cum.plot(ax=ax, color=TEAL, lw=1.5, label="Cumulative Theta (earned)")
        gamma_cum.plot(ax=ax, color=RUST, lw=1.5, label="Cumulative Gamma (paid)")
        cum_full.plot(ax=ax, color=DARK, lw=2, ls="--", label="Net P&L")
        ax.axhline(0, color=GREY, lw=0.8, alpha=0.5)
        ax.set_title("Theta vs. Gamma Decomposition", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Cumulative P&L ($)")
        ax.legend(fontsize=9, framealpha=0.6)
    else:
        df["pnl"].dropna().plot(ax=ax, color=TEAL, lw=1, alpha=0.5)
        ax.set_title("Daily P&L (no theta-gamma decomp available)", fontsize=12, fontweight="bold")

    # Annotate IS/OOS Sharpe on panel 3
    axes[2].text(0.02, 0.92,
                 f"IS Sharpe: {metrics_is['sharpe']:.2f}   OOS Sharpe: {metrics_oos['sharpe']:.2f}",
                 transform=axes[2].transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")


# ============================================================
# MAIN — IMPROVED BACKTEST
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("  VOLATILITY STRATEGY BACKTEST v2 — SPY")
    print("="*60)

    # ── 1. Load data ──
    print("\n[1] Loading data...")
    ohlcv = load_spy_ohlcv(start="2019-01-01")

    # Load VIX via yfinance (fallback if pandas_datareader fails)
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

    # ── 2. Compute vol forecasts ──
    print("\n[2] Computing volatility forecasts...")
    hv20     = close_to_close_vol(ohlcv["close"], window=20)
    yz20     = yang_zhang_vol(ohlcv, window=20)
    ewma     = ewma_vol(ohlcv["close"])

    print("  Fitting GJR-GARCH(1,1,1)...")
    garch    = garch_vol_forecast(ohlcv["close"], use_gjr=True)

    print("  Fitting HAR-RV (rolling OLS)...")
    har      = har_rv_forecast(ohlcv["close"])

    ensemble = combine_forecasts({"ewma": ewma, "garch": garch, "har": har})

    # ── 3. Build master DataFrame ──
    print("\n[3] Building master DataFrame...")
    df = pd.DataFrame({
        "spy_close":    ohlcv["close"],
        "hv_20d":       hv20,
        "yz_vol":       yz20,
        "ewma_vol":     ewma,
        "garch_vol":    garch,
        "har_vol":      har,
        "vol_forecast": ensemble,   # primary forecast
    }).dropna()

    if vix is not None:
        df["vix"] = vix.reindex(df.index, method="ffill")
        df["iv_proxy"] = df["vix"]
    else:
        # Fallback: simulate a realistic VRP (mean-reverting AR(1) around 4%)
        np.random.seed(42)
        n = len(df)
        vrp_sim = np.zeros(n)
        vrp_sim[0] = 0.04
        for t in range(1, n):
            vrp_sim[t] = 0.95 * vrp_sim[t-1] + 0.04*(1-0.95) + np.random.normal(0, 0.008)
        df["iv_proxy"] = df["vol_forecast"] + pd.Series(vrp_sim, index=df.index)

    df = df.dropna()

    # VRP spread (key signal)
    df["vrp_spread"] = df["iv_proxy"] - df["vol_forecast"]

    # Vol-of-vol (regime filter)
    df["vol_of_vol"] = df["vol_forecast"].rolling(20).std()

    # ── 4. Signal generation ──
    print("\n[4] Generating signals (with regime filter)...")
    df["signal"] = generate_signals_v2(
        vrp_spread=df["vrp_spread"],
        vol_of_vol=df["vol_of_vol"],
        threshold=THRESHOLD,
        vov_limit=VOL_OF_VOL_LIMIT,
    )

    signal_counts = df["signal"].value_counts().sort_index()
    print(f"  Signal distribution:  Short={signal_counts.get(-1,0)}  "
          f"Flat={signal_counts.get(0,0)}  Long={signal_counts.get(1,0)}")
    pct_active = (df["signal"] != 0).mean()
    print(f"  % time active: {pct_active:.1%}  (was 98.2% in v1 — now regime-filtered)")

    # ── 5. Dynamic position sizing ──
    print("\n[5] Computing dynamic position sizes...")
    df["pos_size"] = position_size(
        vrp_spread=df["vrp_spread"],
        realized_vol=df["vol_forecast"],
    ) * df["signal"].abs()   # zero out size when signal is flat

    # ── 6. P&L computation (theta-gamma decomposition) ──
    print("\n[6] Computing P&L (theta-gamma decomposition)...")
    pnl_df = compute_pnl_v2(
        signals=df["signal"],
        sizes=df["pos_size"],
        realized_vol=df["vol_forecast"],
        implied_vol=df["iv_proxy"],
        spot=df["spy_close"],
        use_theta_gamma=True,
    )
    df = df.join(pnl_df[["pnl","theta_pnl","gamma_pnl"]], how="left")

    # ── 7. Risk metrics (full sample + IS/OOS split) ──
    print("\n[7] Computing risk metrics...")
    split = pd.Timestamp(IS_END)
    pnl_all = df["pnl"].dropna()
    pnl_is  = pnl_all[pnl_all.index <= split]
    pnl_oos = pnl_all[pnl_all.index >  split]

    metrics_all = compute_risk_metrics(pnl_all, label="Full Sample")
    metrics_is  = compute_risk_metrics(pnl_is,  label="In-Sample (2019–2023)")
    metrics_oos = compute_risk_metrics(pnl_oos, label="Out-of-Sample (2024+)")

    print_risk_report(metrics_all)
    print_risk_report(metrics_is)
    print_risk_report(metrics_oos)

    # ── 8. Save outputs ──
    print("\n[8] Saving outputs...")
    df.to_csv("backtest_v2_results.csv")
    print("  Full results → backtest_v2_results.csv")

    # Risk report CSV
    report_rows = []
    for m in [metrics_all, metrics_is, metrics_oos]:
        report_rows.append({
            "Sample":         m["label"],
            "N Obs":          m["n_obs"],
            "Ann Return":     f"{m['ann_return']:.2%}",
            "Ann Vol":        f"{m['ann_vol']:.2%}",
            "Sharpe":         f"{m['sharpe']:.4f}",
            "Sortino":        f"{m['sortino']:.4f}",
            "Calmar":         f"{m['calmar']:.4f}",
            "VaR 95%":        f"${m['var_95']:.6f}",
            "CVaR 95%":       f"${m['cvar_95']:.6f}",
            "VaR 99%":        f"${m['var_99']:.6f}",
            "CVaR 99%":       f"${m['cvar_99']:.6f}",
            "Max Abs DD":     f"${m['max_abs_dd']:.4f}",
            "Win Rate":       f"{m['win_rate']:.1%}",
            "Profit Factor":  f"{m['profit_factor']:.4f}",
        })
    pd.DataFrame(report_rows).to_csv("backtest_v2_risk_report.csv", index=False)
    print("  Risk report → backtest_v2_risk_report.csv")

    # ── 9. Plot ──
    print("\n[9] Generating chart...")
    plot_results_v2(df, metrics_is, metrics_oos, out_path="backtest_v2_plot.png")

    print("\n[DONE] All outputs saved.")
    print(f"\nKey improvement vs v1:")
    print(f"  v1 Sharpe: -16.27  (sign bug + circular IV)")
    print(f"  v2 Sharpe: {metrics_all['sharpe']:.2f}  (fixed)")
    print(f"  % time active reduced from 98.2% → {pct_active:.1%}")
