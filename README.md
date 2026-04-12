# DTMIF — Volatility engine & VRP backtest

Python package for a **volatility forecasting stack** (Yang–Zhang, EWMA, GJR-GARCH, HAR-RV) and a **SPY volatility risk premium (VRP) backtest** using VIX as an implied-vol proxy.

## Setup

```bash
cd DTMIF_Project
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Editable install (adds `dtmif-backtest` / `dtmif-vol-demo` CLI commands):
pip install -e .
```

Needs **network access** for Yahoo Finance (SPY, VIX) and optional FRED (VIX in the vol demo via `pandas-datareader`).

## Run the backtest (same outputs as before)

From the repo root (outputs are written to the **current working directory**):

```bash
python backtest_v2.py
```

Or:

```bash
python -m dtmif.backtest
```

After `pip install -e .`:

```bash
dtmif-backtest
```

Produces:

- `backtest_v2_results.csv`
- `backtest_v2_risk_report.csv`
- `backtest_v2_plot.png`

## Run the volatility engine demo

```bash
python vol_engine_v2.py
```

Or:

```bash
python -m dtmif.vol_engine
```

Or:

```bash
dtmif-vol-demo
```

Writes `spy_volatility_v2.csv` (and prints a short RMSE table).

## Project layout

| Path | Purpose |
|------|--------|
| `dtmif/vol_engine/` | Data loaders, realized vol, GARCH, HAR-RV, forecast combination, Black–Scholes helpers |
| `dtmif/backtest/` | Config, signals, sizing, P&L, risk metrics, plots, full `run_backtest()` pipeline |
| `backtest_v2.py` | Thin wrapper — re-exports API and runs `run_backtest()` |
| `vol_engine_v2.py` | Thin wrapper — re-exports engine API and runs the vol demo |

New code should import from `dtmif.vol_engine` and `dtmif.backtest` instead of the root shims.

## Configuration

Strategy parameters (thresholds, IS/OOS split date, option tenor for P&L, etc.) live in `dtmif/backtest/config.py`.

## License

MIT (adjust in `pyproject.toml` if you use a different license).
