import pandas as pd

from stocklstm import universe


def _snaps():
    return universe.parse_snapshots(
        pd.DataFrame(
            {
                "date": ["2015-06-01", "2016-03-01", "2017-01-02"],
                "tickers": ["A,B,FB", "A,FB,C", "A,META,C,BRK.B"],
            }
        )
    )


def test_tickers_in_range_includes_member_at_start():
    assert universe.tickers_in_range(_snaps(), "2016-01-01", "2016-12-31") == ["A", "B", "C", "FB"]


def test_membership_matrix_is_point_in_time():
    dates = pd.DatetimeIndex(["2015-01-01", "2016-02-01", "2016-03-01", "2018-01-01"])
    m = universe.membership_matrix(_snaps(), dates)
    assert not m.loc["2015-01-01"].any()
    assert m.loc["2016-02-01", "B"] and not m.loc["2016-03-01", "B"]
    assert m.loc["2016-03-01", "FB"] and not m.loc["2018-01-01", "FB"]
    assert m.loc["2018-01-01", ["A", "META", "C", "BRK.B"]].all()


def test_to_yahoo_applies_alias_and_class_share_format():
    aliases = {"FB": "META"}
    assert universe.to_yahoo("FB", aliases) == "META"
    assert universe.to_yahoo("BRK.B", aliases) == "BRK-B"
    assert universe.to_yahoo("AAPL", aliases) == "AAPL"
