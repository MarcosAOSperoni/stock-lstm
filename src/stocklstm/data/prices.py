import logging
import time

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

FIELDS = ["open", "high", "low", "close", "volume", "raw_close"]


def _download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(
            symbols,
            start=start,
            end=end,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception as exc:  # yfinance raises a variety of network errors
        log.warning("download failed for %d symbols: %s", len(symbols), exc)
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    if not isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_product([symbols, raw.columns])
    long = raw.stack(level=0, future_stack=True).reset_index()
    long.columns = [str(c).lower() for c in long.columns]
    long = long.rename(columns={"level_1": "symbol", "ticker": "symbol"})
    long = long.dropna(subset=["close", "adj close"])
    # "close" is split-adjusted only; keep it as raw_close for dollar volume, since the
    # dividend adjustment in "adj close" depends on dividends paid after the date.
    ratio = long["adj close"] / long["close"]
    long["raw_close"] = long["close"]
    long[["open", "high", "low"]] = long[["open", "high", "low"]].mul(ratio, axis=0)
    long["close"] = long["adj close"]
    return long[["date", "symbol", *FIELDS]]


def download_prices(
    symbols: list[str], start: str, end: str, chunk: int = 100, retries: int = 2
) -> pd.DataFrame:
    """Daily split/dividend-adjusted OHLCV (+ split-only raw_close) in long format.

    Symbols missing from a chunk are retried individually, since Yahoo intermittently
    drops symbols from large batch requests.
    """
    symbols = sorted(set(symbols))
    frames = []
    for i in range(0, len(symbols), chunk):
        part = symbols[i : i + chunk]
        log.info("downloading prices %d-%d of %d", i + 1, i + len(part), len(symbols))
        frames.append(_download(part, start, end))
    got = set().union(*(set(f["symbol"]) for f in frames if not f.empty))
    for attempt in range(retries):
        missing = [s for s in symbols if s not in got]
        if not missing:
            break
        log.info("retrying %d missing symbols (attempt %d)", len(missing), attempt + 1)
        time.sleep(5)
        for sym in missing:
            df = _download([sym], start, end)
            if not df.empty:
                frames.append(df)
                got.add(sym)
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out.sort_values(["symbol", "date"]).reset_index(drop=True)
