# stock-lstm

LSTM stock predictor for the S&P 500 (point-in-time universe) with a harness for evaluating
alternative data signals. Research project, not trading advice.

## Status

Phase 1: data pipeline, walk-forward harness and baselines. The LSTM (phase 2) and the
alternative-data plugins (phase 3) are not implemented yet.

## Setup

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
stocklstm download                    # membership, prices, market series, FRED -> data/raw
stocklstm build-dataset               # features + labels -> data/processed/dataset.parquet
stocklstm baselines                   # walk-forward baselines -> reports/baselines/
stocklstm run                         # all three
stocklstm run --max-tickers 100 --horizons 1d,1m   # quick subset
```

Or `scripts/run_baselines.sh`, which creates a venv and runs everything. Set
`STOCKLSTM_DATA_DIR` to put data elsewhere (e.g. an NFS share). Settings live in
`configs/default.yaml`.

## Design

- **Universe:** historical S&P 500 membership snapshots from
  [fja05680/sp500](https://github.com/fja05680/sp500). A stock is only used on dates it was in
  the index. Renamed tickers are mapped in `configs/ticker_aliases.yaml`.
- **Data:** adjusted daily OHLCV from Yahoo Finance (yfinance); `^GSPC`, `^VIX`, oil, the dollar
  index, gold and sector ETFs; FRED rates and credit spreads (lagged one trading day).
- **Features:** price/volume features (returns, 12-1 momentum, volatility, RSI, MACD, distance from
  52-week high, volume z-score, dollar volume, beta), ranked cross-sectionally per date, plus
  market context. Each feature only uses data up to that date.
- **Target:** for horizons 1d/1w/1m/1y, the forward log return from `t+1` to `t+1+h` minus the
  S&P 500's, converted to a centered percentile rank across members on date `t`.
- **Validation:** expanding-window walk-forward (3-year minimum train, 6-month test), with a purge
  gap of `horizon + execution_lag` days. The last 18 months are held out and not used.
- **Baselines:** 12-1 momentum, 5-day reversal, LightGBM on all features.
- **Metrics:** daily Spearman IC (mean, IR, t-stat), hit rate, and a top-minus-bottom decile
  long/short backtest with 10 bps cost per unit turnover, rebalanced every `h` days.

## Known data limitations

- yfinance has no history for some former members (mostly acquired or bankrupt companies). The
  share of member-days with prices is logged and written to `data/raw/coverage.csv`. Missing
  delisted stocks bias results (survivorship bias), so treat backtest returns as optimistic.
- The 1-year horizon has only a handful of independent periods; its metrics are noisy.
