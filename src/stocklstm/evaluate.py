import numpy as np
import pandas as pd


def daily_ic(df: pd.DataFrame, pred: str, label: str) -> pd.Series:
    """Spearman rank correlation between prediction and label, per date."""
    d = df[["date", pred, label]].dropna()
    g = d.groupby("date")
    r = pd.DataFrame({"p": g[pred].rank(), "y": g[label].rank(), "date": d["date"]}, index=d.index)
    return r.groupby("date").apply(lambda x: x["p"].corr(x["y"]) if len(x) > 2 else np.nan)


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """IC stats; the t-stat uses n_days / horizon effective observations (labels overlap)."""
    ic = ic.dropna()
    n = len(ic) / horizon
    std = ic.std()
    return {
        "mean_ic": ic.mean(),
        "ic_std": ic.std(),
        "ic_ir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "ic_t_stat": ic.mean() / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan,
        "pct_ic_positive": (ic > 0).mean(),
        "n_days": len(ic),
    }


def hit_rate(df: pd.DataFrame, pred: str, label: str) -> float:
    """How often the prediction lands on the correct side of the cross-sectional median."""
    d = df[["date", pred, label]].dropna()
    centered = d[pred] - d.groupby("date")[pred].transform("median")
    mask = (centered != 0) & (d[label] != 0)
    return float((np.sign(centered[mask]) == np.sign(d.loc[mask, label])).mean())


def _weights(names: pd.Index, sign: float) -> pd.Series:
    return pd.Series(sign / len(names), index=names) if len(names) else pd.Series(dtype=float)


def long_short_backtest(
    df: pd.DataFrame, pred: str, fwd: str, horizon: int, quantile: float, cost_bps: float
) -> dict:
    """Equal-weight top-minus-bottom quantile portfolio, rebalanced every `horizon` days.

    `fwd` holds forward log excess returns over the holding period. Costs are charged on
    one-way turnover of the combined long/short book.
    """
    d = df[["date", "ticker", pred, fwd]].dropna()
    dates = np.sort(d["date"].unique())[::horizon]
    d = d[d["date"].isin(dates)]
    prev_long = prev_short = pd.Series(dtype=float)
    rows = []
    for date, g in d.groupby("date"):
        g = g.set_index("ticker")
        lo, hi = g[pred].quantile([quantile, 1 - quantile])
        top, bot = g.index[g[pred] >= hi], g.index[g[pred] <= lo]
        if len(top) == 0 or len(bot) == 0:
            continue
        simple = np.expm1(g[fwd])
        w_long, w_short = _weights(top, 1.0), _weights(bot, -1.0)
        turnover = (
            w_long.sub(prev_long, fill_value=0).abs().sum()
            + w_short.sub(prev_short, fill_value=0).abs().sum()
        )
        long_turn = w_long.sub(prev_long, fill_value=0).abs().sum()
        cost = cost_bps / 1e4
        rows.append(
            {
                "date": date,
                "long_short_gross": simple[top].mean() - simple[bot].mean(),
                "long_short": simple[top].mean() - simple[bot].mean() - turnover * cost,
                "long_only": simple[top].mean() - long_turn * cost,
                "turnover": turnover / 2,
            }
        )
        prev_long, prev_short = w_long, w_short
    res = pd.DataFrame(rows).set_index("date")
    ppy = 252 / horizon
    out = {"n_periods": len(res), "avg_turnover": res["turnover"].mean() if len(res) else np.nan}
    for col in ("long_short", "long_only"):
        r = res[col] if len(res) else pd.Series(dtype=float)
        equity = (1 + r).cumprod()
        out[f"{col}_ann_return"] = r.mean() * ppy
        out[f"{col}_sharpe"] = r.mean() / r.std() * np.sqrt(ppy) if r.std() > 0 else np.nan
        out[f"{col}_max_drawdown"] = (equity / equity.cummax() - 1).min() if len(r) else np.nan
    gross = res["long_short_gross"].mean() if len(res) else np.nan
    out["long_short_gross_ann_return"] = gross * ppy
    return out
