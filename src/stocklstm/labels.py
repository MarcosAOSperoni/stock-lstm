import pandas as pd

from .features import safe_log


def forward_log_return(close, horizon: int, lag: int):
    """Log return from close[t+lag] to close[t+lag+horizon] (DataFrame or Series)."""
    logc = safe_log(close)
    return logc.shift(-(lag + horizon)) - logc.shift(-lag)


def forward_excess_return(
    close: pd.DataFrame, bench: pd.Series, horizon: int, lag: int
) -> pd.DataFrame:
    """Forward log return minus the benchmark's over the same window."""
    return forward_log_return(close, horizon, lag).sub(
        forward_log_return(bench, horizon, lag), axis=0
    )


def rank_label(excess: pd.DataFrame, member: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank among members on the signal date, centered on 0."""
    return excess.where(member).rank(axis=1, pct=True) - 0.5
