import pandas as pd

from stocklstm.splits import walk_forward_folds


def test_folds_are_purged_and_stay_before_holdout():
    dates = pd.bdate_range("2016-01-01", "2024-12-31")
    holdout = pd.Timestamp("2023-06-01")
    purge = 22
    folds = walk_forward_folds(dates, "2016-01-01", holdout, 3, 6, purge)
    assert len(folds) == 9
    pos = {d: i for i, d in enumerate(dates)}
    for prev, fold in zip(folds, folds[1:], strict=False):
        assert fold.test_start > prev.test_end
    for fold in folds:
        assert fold.train_start == dates[0]
        assert pos[fold.test_start] - pos[fold.train_end] == purge + 1
    assert pos[holdout] - pos[folds[-1].test_end] > purge
