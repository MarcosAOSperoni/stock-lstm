import pandas as pd

from .features import safe_log


def forward_excess_return(
    close: pd.DataFrame, bench: pd.Series, horizon: int, lag: int
) -> pd.DataFrame:
    """Log return from close[t+lag] to close[t+lag+horizon], minus the benchmark's."""
    logc = safe_log(close)
    logb = safe_log(bench)
    stock = logc.shift(-(lag + horizon)) - logc.shift(-lag)
    market = logb.shift(-(lag + horizon)) - logb.shift(-lag)
    return stock.sub(market, axis=0)


def rank_label(excess: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank among members on the signal date, centered on 0."""
    return excess.where(member).rank(axis=1, pct=True) - 0.5
