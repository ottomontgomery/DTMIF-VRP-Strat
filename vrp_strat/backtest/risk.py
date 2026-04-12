"""Risk metrics and reporting."""

import numpy as np
import pandas as pd


def compute_risk_metrics(pnl: pd.Series, label: str = "Strategy") -> dict:
    """Risk metrics for daily **fractional** P&L (e.g. spot-normalized returns)."""
    clean = pnl.dropna()
    n = len(clean)

    sharpe = clean.mean() / clean.std() * np.sqrt(252) if clean.std() > 0 else np.nan

    var95 = -np.percentile(clean, 5)
    var99 = -np.percentile(clean, 1)
    cvar95 = -clean[clean <= -var95].mean()
    cvar99 = -clean[clean <= -var99].mean()

    ann_return = clean.mean() * 252
    cum = clean.cumsum()
    peak = cum.cummax()
    dd = cum - peak

    max_abs_dd = dd.min()

    # Cumulative equity is sum of fractional daily P&L; peak-relative % from a
    # synthetic zero start produced misleading extremes when equity stayed ≤ 0.
    pct_dd = max_abs_dd

    calmar = ann_return / abs(max_abs_dd) if max_abs_dd != 0 else np.nan

    downside = clean[clean < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else np.nan

    wins = clean[clean > 0]
    losses = clean[clean < 0]
    win_rate = len(wins) / n
    pf = wins.sum() / losses.abs().sum() if len(losses) > 0 else np.inf

    return {
        "label": label,
        "n_obs": n,
        "ann_return": ann_return,
        "ann_vol": clean.std() * np.sqrt(252),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "var_95": var95,
        "cvar_95": cvar95,
        "var_99": var99,
        "cvar_99": cvar99,
        "max_abs_dd": max_abs_dd,
        "max_pct_dd": pct_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "cum_pnl": cum,
        "drawdown": dd,
    }


def print_risk_report(m: dict) -> None:
    """Print ``m`` as a formatted table."""
    w = 55
    print(f"\n{'='*w}")
    print(f"  RISK REPORT: {m['label']}")
    print(f"{'='*w}")
    rows = [
        ("Obs (trading days)", f"{m['n_obs']}"),
        ("Annualised Return", f"{m['ann_return']:.2%}"),
        ("Annualised Vol", f"{m['ann_vol']:.2%}"),
        ("Sharpe Ratio (ann.)", f"{m['sharpe']:.4f}"),
        ("Sortino Ratio (ann.)", f"{m['sortino']:.4f}"),
        ("Calmar Ratio", f"{m['calmar']:.4f}"),
        ("VaR 95% (daily, notional)", f"{m['var_95']:.4%}"),
        ("CVaR 95% (daily, notional)", f"{m['cvar_95']:.4%}"),
        ("VaR 99% (daily, notional)", f"{m['var_99']:.4%}"),
        ("CVaR 99% (daily, notional)", f"{m['cvar_99']:.4%}"),
        ("Max drawdown (cum. return)", f"{m['max_abs_dd']:.2%}"),
        ("Win Rate", f"{m['win_rate']:.1%}"),
        ("Profit Factor", f"{m['profit_factor']:.4f}"),
    ]
    for k, v in rows:
        print(f"  {k:<35} {v}")
    print(f"{'='*w}\n")
