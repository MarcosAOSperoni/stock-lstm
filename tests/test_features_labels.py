import numpy as np
import pandas as pd

from stocklstm.features import cross_sectional_rank, stock_features
from stocklstm.labels import forward_excess_return, rank_label


def test_features_do_not_look_ahead(wide_prices):
    close, volume, bench, _ = wide_prices
    cut = 300
    base = stock_features(close, volume, bench)
    shocked_close = close.copy()
    shocked_close.iloc[cut + 1 :] *= 3.0
    shocked_bench = bench.copy()
    shocked_bench.iloc[cut + 1 :] *= 0.5
    shocked = stock_features(shocked_close, volume, shocked_bench)
    for name in base:
        pd.testing.assert_frame_equal(
            base[name].iloc[: cut + 1], shocked[name].iloc[: cut + 1], check_exact=False
        )


def test_forward_excess_return_matches_definition(wide_prices):
    close, _, bench, _ = wide_prices
    h, lag, t = 5, 1, 100
    ex = forward_excess_return(close, bench, h, lag)
    expected = np.log(close.iloc[t + lag + h] / close.iloc[t + lag]) - np.log(
        bench.iloc[t + lag + h] / bench.iloc[t + lag]
    )
    np.testing.assert_allclose(ex.iloc[t].to_numpy(), expected.to_numpy())
    assert ex.iloc[-(h + lag) :].isna().all().all()


def test_ranks_only_include_members(wide_prices):
    close, _, bench, member = wide_prices
    ex = forward_excess_return(close, bench, 5, 1)
    y = rank_label(ex, member)
    assert y.iloc[:200, :5].isna().all().all()
    row = y.iloc[100].dropna()
    assert len(row) == 15
    assert row.min() > -0.5 and row.max() == 0.5
    feat = cross_sectional_rank(close, member)
    assert feat.iloc[100, :5].isna().all()
