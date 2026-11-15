import numpy as np
import pandas as pd

from stocklstm.evaluate import daily_ic, hit_rate, ic_summary, long_short_backtest


def _panel(noise: float, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=120)
    rows = []
    for d in dates:
        fwd = rng.normal(0, 0.02, 50)
        rows.append(
            pd.DataFrame(
                {
                    "date": d,
                    "ticker": [f"T{i}" for i in range(50)],
                    "fwd": fwd,
                    "y": pd.Series(fwd).rank(pct=True) - 0.5,
                    "pred": fwd + rng.normal(0, noise, 50),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_perfect_prediction_scores_perfectly():
    df = _panel(noise=0.0)
    ic = daily_ic(df, "pred", "y")
    assert np.allclose(ic, 1.0)
    assert hit_rate(df, "pred", "y") == 1.0
    assert np.isclose(ic_summary(ic, 5)["mean_ic"], 1.0)


def test_random_prediction_has_no_skill():
    df = _panel(noise=0.0)
    df["pred"] = np.random.default_rng(1).normal(size=len(df))
    assert abs(ic_summary(daily_ic(df, "pred", "y"), 1)["mean_ic"]) < 0.05


def test_costs_reduce_long_short_returns():
    df = _panel(noise=0.01)
    free = long_short_backtest(df, "pred", "fwd", 5, 0.1, 0)
    costly = long_short_backtest(df, "pred", "fwd", 5, 0.1, 50)
    assert free["n_periods"] == 24
    assert free["long_short_ann_return"] > costly["long_short_ann_return"] > 0


def _fixed_book(fwd_by_day: list[dict[str, float]], bench: float = 0.0) -> pd.DataFrame:
    """Two longs (A, B) and two shorts (C, D) every period; `fwd` per ticker per period."""
    rows = []
    for i, fwd in enumerate(fwd_by_day):
        for t, pred in zip("ABCD", [2.0, 2.0, -2.0, -2.0], strict=True):
            rows.append(
                {
                    "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
                    "ticker": t,
                    "pred": pred,
                    "fwd": fwd.get(t, 0.0),
                    "fwd_bench": bench,
                }
            )
    return pd.DataFrame(rows)


def test_long_short_uses_stock_returns_not_benchmark_ratios():
    bench = np.log(1.2)
    df = _fixed_book([{"A": 0.1, "B": 0.1}], bench=bench)
    res = long_short_backtest(df, "pred", "fwd", 1, 0.5, 0, bench="fwd_bench")
    # A, B return 1.2 * e^0.1 - 1; C, D return exactly the benchmark (20%).
    expected = 1.2 * np.expm1(0.1)
    assert np.isclose(res["long_short_gross_ann_return"], expected * 252)


def test_rebalancing_drifted_weights_costs_money():
    df = _fixed_book([{"A": np.log(1.5)}, {}])
    free = long_short_backtest(df, "pred", "fwd", 1, 0.5, 0)
    # Opening the book trades 1.0 one-way; A drifts to 60/40 vs B, so rebalancing trades 0.1.
    assert np.isclose(free["avg_turnover"], (1.0 + 0.1) / 2)


def test_first_period_loss_counts_in_drawdown():
    df = _fixed_book([{"A": np.log(0.8), "B": np.log(0.8)}, {}])
    res = long_short_backtest(df, "pred", "fwd", 1, 0.5, 0)
    assert np.isclose(res["long_short_max_drawdown"], -0.2)
