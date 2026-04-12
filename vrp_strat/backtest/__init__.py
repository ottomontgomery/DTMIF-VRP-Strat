"""SPY volatility risk premium backtest (v2)."""

from vrp_strat.backtest.config import (
    IS_END,
    LAMBDA_EWMA,
    MAX_POSITION,
    RISK_FREE_RATE,
    THRESHOLD,
    T_OPTION,
    VOL_OF_VOL_LIMIT,
    VRP_MIN_SIGNAL,
)
from vrp_strat.backtest.pnl import compute_pnl_v2
from vrp_strat.backtest.pipeline import run_backtest
from vrp_strat.backtest.plotting import plot_results_v2
from vrp_strat.backtest.position_sizing import position_size
from vrp_strat.backtest.risk import compute_risk_metrics, print_risk_report
from vrp_strat.backtest.signals import generate_signals_v2

__all__ = [
    "IS_END",
    "LAMBDA_EWMA",
    "MAX_POSITION",
    "RISK_FREE_RATE",
    "THRESHOLD",
    "T_OPTION",
    "VOL_OF_VOL_LIMIT",
    "VRP_MIN_SIGNAL",
    "compute_pnl_v2",
    "compute_risk_metrics",
    "generate_signals_v2",
    "plot_results_v2",
    "position_size",
    "print_risk_report",
    "run_backtest",
]
