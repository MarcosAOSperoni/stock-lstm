"""Features computed on wide frames (dates x tickers) using only data up to each date."""

import numpy as np
import pandas as pd

RETURN_WINDOWS = (1, 5, 21, 63, 126, 252)


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, min_periods=window).mean()
    return 100 - 100 / (1 + gain / loss.replace(0, np.nan))


def safe_log(x):
    """Log of strictly positive values; zero/negative prices (e.g. CL=F in 2020-04) become NaN."""
    return np.log(x.where(x > 0))


def stock_features(
    close: pd.DataFrame, volume: pd.DataFrame, bench: pd.Series, raw_close: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Raw per-stock price/volume features, each a wide frame aligned to `close`.

    `close` is total-return adjusted; `raw_close` is split-adjusted only and is used for
    dollar volume so the feature doesn't depend on later dividends.
    """
    logc = safe_log(close)
    lr = logc.diff()
    mlr = safe_log(bench).diff()
    f: dict[str, pd.DataFrame] = {}
    for k in RETURN_WINDOWS:
        f[f"ret_{k}"] = logc - logc.shift(k)
    f["mom_12_1"] = logc.shift(21) - logc.shift(252)
    for k in (21, 63):
        f[f"vol_{k}"] = lr.rolling(k, min_periods=int(0.8 * k)).std()
    f["rsi_14"] = _rsi(close)
    ema12 = close.ewm(span=12, min_periods=26).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    f["macd"] = (ema12 - ema26) / close
    f["dist_52w_high"] = close / close.rolling(252, min_periods=200).max() - 1
    logv = np.log1p(volume)
    f["volume_z_63"] = (logv - logv.rolling(63, min_periods=50).mean()) / logv.rolling(
        63, min_periods=50
    ).std()
    f["log_dollar_vol_21"] = safe_log((raw_close * volume).rolling(21, min_periods=15).mean())
    cov = lr.mul(mlr, axis=0).rolling(63, min_periods=50).mean() - lr.rolling(
        63, min_periods=50
    ).mean().mul(mlr.rolling(63, min_periods=50).mean(), axis=0)
    f["beta_63"] = cov.div(mlr.rolling(63, min_periods=50).var(ddof=0), axis=0)
    return f


def cross_sectional_rank(frame: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank among index members on each date, centered on 0 (range ~[-0.5, 0.5])."""
    return frame.where(member).rank(axis=1, pct=True) - 0.5


def market_features(
    market_close: pd.DataFrame,
    fred: pd.DataFrame,
    dates: pd.DatetimeIndex,
    benchmark: str,
    series: dict[str, str],
    sector_etfs: list[str],
    fred_lag_days: int,
) -> pd.DataFrame:
    """Date-level context shared by all stocks: index trend/vol, VIX, commodities, rates."""
    mc = market_close.reindex(dates).ffill(limit=3)
    logm = safe_log(mc)
    b = logm[benchmark]
    out = pd.DataFrame(index=dates)
    for k in (1, 5, 21, 63):
        out[f"mkt_ret_{k}"] = b - b.shift(k)
    out["mkt_vol_21"] = b.diff().rolling(21, min_periods=15).std()
    for name, sym in series.items():
        if sym not in mc:
            continue
        if name == "vix":
            out["vix"] = mc[sym]
            out["vix_chg_5"] = mc[sym] - mc[sym].shift(5)
        else:
            out[f"{name}_ret_21"] = logm[sym] - logm[sym].shift(21)
    for etf in sector_etfs:
        if etf in logm:
            rel = (logm[etf] - logm[etf].shift(21)) - (b - b.shift(21))
            out[f"sector_{etf.lower()}_rel_21"] = rel
    fr = fred.reindex(fred.index.union(dates)).ffill().reindex(dates).shift(fred_lag_days)
    for col in fr:
        out[col] = fr[col]
        out[f"{col}_chg_21"] = fr[col] - fr[col].shift(21)
    return out
