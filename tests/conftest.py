import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def wide_prices():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    tickers = [f"T{i}" for i in range(20)]
    rets = rng.normal(0.0003, 0.02, size=(len(dates), len(tickers)))
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)
    volume = pd.DataFrame(
        rng.integers(1e5, 1e6, size=close.shape).astype(float), index=dates, columns=tickers
    )
    bench = close.mean(axis=1)
    member = pd.DataFrame(True, index=dates, columns=tickers)
    member.iloc[:200, :5] = False
    return close, volume, bench, member
