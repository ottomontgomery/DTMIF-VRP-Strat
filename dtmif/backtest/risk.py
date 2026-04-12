"""Risk metrics and reporting."""

import numpy as np
import pandas as pd


def compute_risk_metrics(pnl: pd.Series, label: str = "Strategy") -> dict:
    """Risk metrics from a daily P&L series."""
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

    cum0 = pd.concat([pd.Series([0.0]), cum])
    peak0 = cum0.cummax()
    dd0 = cum0 - peak0
    peak0_pos = peak0[peak0 > 0]
    if len(peak0_pos) > 0:
        pct_dd = (dd0[peak0_pos.index] / peak0_pos).min()
    else:
        pct_dd = np.nan

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
        ("VaR 95% (daily $)", f"${m['var_95']:.6f}"),
        ("CVaR 95% (daily $)", f"${m['cvar_95']:.6f}"),
        ("VaR 99% (daily $)", f"${m['var_99']:.6f}"),
        ("CVaR 99% (daily $)", f"${m['cvar_99']:.6f}"),
        ("Max Abs Drawdown ($)", f"${m['max_abs_dd']:.4f}"),
        (
            "Max % Drawdown",
            f"{m['max_pct_dd']:.2%}" if m["max_pct_dd"] is not np.nan else "N/A (never in profit)",
        ),
        ("Win Rate", f"{m['win_rate']:.1%}"),
        ("Profit Factor", f"{m['profit_factor']:.4f}"),
    ]
    for k, v in rows:
        print(f"  {k:<35} {v}")
    print(f"{'='*w}\n")
