"""Point-in-time S&P 500 membership.

The membership source is a table of snapshots: one row per date on which the index
changed, listing every member as of that date (historical symbols, e.g. FB before META).
"""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml


def download_snapshots(url: str) -> pd.DataFrame:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return parse_snapshots(pd.read_csv(StringIO(resp.text)))


def parse_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["date", "tickers"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


def _snapshot_sets(snapshots: pd.DataFrame) -> list[frozenset[str]]:
    return [frozenset(t.strip() for t in s.split(",") if t.strip()) for s in snapshots["tickers"]]


def tickers_in_range(snapshots: pd.DataFrame, start, end) -> list[str]:
    """Every symbol that was an index member at any point in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    dates = snapshots["date"].to_numpy()
    sets = _snapshot_sets(snapshots)
    first = max(int(np.searchsorted(dates, np.datetime64(start), side="right")) - 1, 0)
    last = int(np.searchsorted(dates, np.datetime64(end), side="right"))
    out: set[str] = set()
    for s in sets[first:last]:
        out |= s
    return sorted(out)


def membership_matrix(snapshots: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean frame (dates x symbols): True where the symbol was a member on that date."""
    sets = _snapshot_sets(snapshots)
    idx = np.searchsorted(snapshots["date"].to_numpy(), dates.to_numpy(), side="right") - 1
    used = sorted(set().union(*(sets[i] for i in set(idx) if i >= 0)))
    col = {t: j for j, t in enumerate(used)}
    mat = np.zeros((len(dates), len(used)), dtype=bool)
    for row, i in enumerate(idx):
        if i >= 0:
            mat[row, [col[t] for t in sets[i]]] = True
    return pd.DataFrame(mat, index=dates, columns=used)


def load_aliases(path: str | Path | None) -> dict[str, str]:
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def to_yahoo(symbol: str, aliases: dict[str, str]) -> str:
    """Map a historical index symbol to the Yahoo Finance symbol holding its price history."""
    return aliases.get(symbol, symbol).replace(".", "-")
