"""Backtest diagnostic charts."""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

matplotlib.use("Agg")

from vrp_strat.backtest.config import IS_END, THRESHOLD, VOL_OF_VOL_LIMIT


def plot_results_v2(
    df: pd.DataFrame,
    metrics_is: dict,
    metrics_oos: dict,
    out_path: str = "backtest_v2_plot.png",
) -> None:
    """Write the four-panel backtest figure to ``out_path``."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=False)
    fig.patch.set_facecolor("#F7F6F2")
    for ax in axes:
        ax.set_facecolor("#F9F8F5")
        ax.spines[["top", "right"]].set_visible(False)

    TEAL = "#20808D"
    RUST = "#A84B2F"
    DARK = "#0A2540"
    GOLD = "#E8B84B"
    RED = "#A13544"
    GREY = "#7A7974"

    ax = axes[0]
    df["vix"].plot(ax=ax, color=RUST, lw=1.4, ls="--", label="VIX (IV proxy)")
    df["vol_forecast"].plot(ax=ax, color=TEAL, lw=1.6, label="Vol Forecast (GJR-GARCH)")
    df["hv_20d"].plot(ax=ax, color=DARK, lw=1.0, alpha=0.6, label="HV-20 (realized)")
    ax.set_title(
        "Volatility Comparison — VIX vs. Forecast vs. Realized",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.set_ylabel("Annualised Vol")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    ax = axes[1]
    df["vrp_spread"].plot(ax=ax, color=GOLD, lw=1.5, label="VRP Spread (VIX − Forecast)")
    ax.axhline(THRESHOLD, color=RED, ls="--", lw=1.2, alpha=0.7, label="+Threshold")
    ax.axhline(-THRESHOLD, color=TEAL, ls="--", lw=1.2, alpha=0.7, label="−Threshold")
    ax.axhline(0, color=GREY, lw=0.8, alpha=0.5)

    blocked = df["vol_of_vol"] > VOL_OF_VOL_LIMIT
    if blocked.any():
        ax.fill_between(
            df.index,
            ax.get_ylim()[0] if ax.get_ylim()[0] < -0.1 else -0.15,
            0.15,
            where=blocked,
            alpha=0.12,
            color=RED,
            label="Regime filter active",
        )

    ax.set_title("VRP Spread with Threshold and Regime Filter", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Spread (IV − Forecast)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, framealpha=0.6)

    ax = axes[2]
    split = pd.Timestamp(IS_END)
    cum_full = df["pnl"].dropna().cumsum()

    is_mask = cum_full.index <= split
    oos_mask = cum_full.index > split

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

    axes[2].text(
        0.02,
        0.92,
        f"IS Sharpe: {metrics_is['sharpe']:.2f}   OOS Sharpe: {metrics_oos['sharpe']:.2f}",
        transform=axes[2].transAxes,
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout(pad=2.0)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved → {out_path}")
