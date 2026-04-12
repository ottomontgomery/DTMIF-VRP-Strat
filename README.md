# vrp_strat — Volatility engine & VRP backtest

Python package for a **volatility forecasting stack** (Yang–Zhang, EWMA, GJR-GARCH, HAR-RV) and a **SPY volatility risk premium (VRP) backtest** using VIX as an implied-vol proxy.

## Setup

```bash
cd VRP_Project_DTMIF   # or your clone path
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Editable install (adds `vrp-backtest` / `vrp-vol-demo` CLI commands):
pip install -e .
```

Needs **network access** for Yahoo Finance (SPY, VIX) and optional FRED (VIX in the vol demo via `pandas-datareader`).

## Run the backtest (same outputs as before)

From the repo root (outputs are written to the **current working directory**):

```bash
python -m vrp_strat.backtest
```

After `pip install -e .`:

```bash
vrp-backtest
```

Produces:

- `backtest_v2_results.csv`
- `backtest_v2_risk_report.csv`
- `backtest_v2_plot.png`

## Run the volatility engine demo

```bash
python -m vrp_strat.vol_engine
```

Or:

```bash
vrp-vol-demo
```

Writes `spy_volatility_v2.csv` (and prints a short RMSE table).

## Project layout

| Path | Purpose |
|------|--------|
| `vrp_strat/vol_engine/` | Data loaders, realized vol, GARCH, HAR-RV, forecast combination, Black–Scholes helpers |
| `vrp_strat/backtest/` | Config, signals, sizing, P&L, risk metrics, plots, full `run_backtest()` pipeline |

Import from `vrp_strat.vol_engine` and `vrp_strat.backtest`.

## Configuration

Strategy parameters (thresholds, IS/OOS split date, option tenor for P&L, etc.) live in `vrp_strat/backtest/config.py`.

## License

MIT (adjust in `pyproject.toml` if you use a different license).
