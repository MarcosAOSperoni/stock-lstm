"""Raw data download and assembly of the modelling table (one row per member-day)."""

import logging

import numpy as np
import pandas as pd

from . import universe
from .config import processed_dir, raw_dir
from .data.macro import download_fred
from .data.prices import download_prices
from .features import cross_sectional_rank, market_features, stock_features
from .labels import forward_excess_return, forward_log_return, rank_label

log = logging.getLogger(__name__)

ID_COLS = ["date", "ticker"]


def download_all(cfg: dict) -> pd.DataFrame:
    raw = raw_dir(cfg)
    end = (pd.Timestamp(cfg["end_date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    snaps = universe.download_snapshots(cfg["universe"]["membership_url"])
    snaps.to_csv(raw / "membership.csv", index=False)
    aliases = universe.load_aliases(cfg["universe"].get("aliases_file"))
    symbols = universe.tickers_in_range(snaps, cfg["start_date"], cfg["end_date"])
    yahoo = {s: universe.to_yahoo(s, aliases) for s in symbols}
    log.info("%d historical members since %s", len(symbols), cfg["start_date"])

    px = download_prices(sorted(set(yahoo.values())), cfg["history_start"], end)
    by_symbol = dict(tuple(px.groupby("symbol")))
    prices = pd.concat(
        [by_symbol[y].assign(ticker=s) for s, y in yahoo.items() if y in by_symbol],
        ignore_index=True,
    ).drop(columns="symbol")
    prices.to_parquet(raw / "prices.parquet", index=False)

    m = cfg["market"]
    market = download_prices(
        [m["benchmark"], *m["series"].values(), *m["sector_etfs"]], cfg["history_start"], end
    )
    market.to_parquet(raw / "market.parquet", index=False)

    fred = download_fred(cfg["fred"]["series"], cfg["history_start"])
    fred.to_parquet(raw / "fred.parquet")

    calendar = market.loc[market["symbol"] == m["benchmark"], "date"]
    cov = coverage_report(snaps, prices, pd.DatetimeIndex(calendar), cfg["start_date"])
    cov.to_csv(raw / "coverage.csv")
    log.info("coverage: %s", coverage_summary(cov))
    return cov


def coverage_report(
    snaps: pd.DataFrame, prices: pd.DataFrame, dates: pd.DatetimeIndex, start
) -> pd.DataFrame:
    """Per historical member: index-member days vs member days with a price."""
    dates = pd.DatetimeIndex(np.sort(dates.unique()))
    dates = dates[dates >= pd.Timestamp(start)]
    member = universe.membership_matrix(snaps, dates)
    close = prices.pivot(index="date", columns="ticker", values="close").reindex(
        index=dates, columns=member.columns
    )
    out = pd.DataFrame({"member_days": member.sum(), "priced_days": (member & close.notna()).sum()})
    out = out[out["member_days"] > 0]
    out["coverage"] = out["priced_days"] / out["member_days"]
    return out.sort_values(["coverage", "member_days"])


def coverage_summary(cov: pd.DataFrame) -> dict:
    return {
        "historical_members": len(cov),
        "members_without_prices": int((cov["priced_days"] == 0).sum()),
        "member_day_coverage": float(cov["priced_days"].sum() / cov["member_days"].sum()),
    }


def load_raw(cfg: dict):
    raw = raw_dir(cfg)
    snaps = universe.parse_snapshots(pd.read_csv(raw / "membership.csv"))
    prices = pd.read_parquet(raw / "prices.parquet")
    market = pd.read_parquet(raw / "market.parquet")
    fred = pd.read_parquet(raw / "fred.parquet")
    return snaps, prices, market, fred


def build_dataset(cfg: dict, max_tickers: int | None = None, seed: int = 0) -> pd.DataFrame:
    snaps, prices, market, fred = load_raw(cfg)
    m = cfg["market"]
    mclose = market.pivot(index="date", columns="symbol", values="close")
    dates = mclose[m["benchmark"]].dropna().index
    dates = dates[dates >= pd.Timestamp(cfg["history_start"])]
    bench = mclose[m["benchmark"]].reindex(dates)

    close = prices.pivot(index="date", columns="ticker", values="close").reindex(dates)
    volume = prices.pivot(index="date", columns="ticker", values="volume").reindex(dates)
    raw_close = prices.pivot(index="date", columns="ticker", values="raw_close").reindex(dates)
    if max_tickers and max_tickers < close.shape[1]:
        rng = np.random.default_rng(seed)
        keep = sorted(rng.choice(close.columns, size=max_tickers, replace=False))
        close, volume, raw_close = close[keep], volume[keep], raw_close[keep]
    member = universe.membership_matrix(snaps, dates).reindex(
        columns=close.columns, fill_value=False
    )

    cols: dict[str, pd.DataFrame] = {}
    for name, frame in stock_features(close, volume, bench, raw_close).items():
        cols[f"cs_{name}"] = cross_sectional_rank(frame, member)
    for name, h in cfg["horizons"].items():
        excess = forward_excess_return(close, bench, h, cfg["execution_lag"])
        cols[f"fwd_{name}"] = excess
        cols[f"y_{name}"] = rank_label(excess, member)

    mkt = market_features(
        mclose,
        fred,
        dates,
        m["benchmark"],
        m["series"],
        m["sector_etfs"],
        cfg["fred"]["lag_days"],
    )
    for name, h in cfg["horizons"].items():
        mkt[f"fwd_bench_{name}"] = forward_log_return(bench, h, cfg["execution_lag"])

    in_study = np.asarray(dates >= pd.Timestamp(cfg["start_date"]))[:, None]
    r, c = np.nonzero(member.to_numpy() & close.notna().to_numpy() & in_study)
    out = pd.DataFrame({"date": dates[r], "ticker": close.columns[c]})
    for name, frame in cols.items():
        out[name] = frame.to_numpy(dtype=np.float32)[r, c]
    mk = mkt.to_numpy(dtype=np.float32)[r]
    for j, name in enumerate(mkt.columns):
        out[name] = mk[:, j]
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c not in ID_COLS and not c.startswith("fwd_") and not c.startswith("y_")
    ]


def dataset_path(cfg: dict):
    return processed_dir(cfg) / "dataset.parquet"
