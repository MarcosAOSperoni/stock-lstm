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
