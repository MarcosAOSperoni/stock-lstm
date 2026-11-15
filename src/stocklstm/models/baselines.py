import lightgbm as lgb
import numpy as np
import pandas as pd

# name -> (feature column, sign). Classic factors that need no training.
FACTOR_BASELINES = {
    "momentum_12_1": ("cs_mom_12_1", 1.0),
    "reversal_5d": ("cs_ret_5", -1.0),
}


def factor_predictions(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {name: sign * df[col] for name, (col, sign) in FACTOR_BASELINES.items()},
        index=df.index,
    )


def fit_lightgbm(
    train: pd.DataFrame, features: list[str], label: str, params: dict, rounds: int
) -> lgb.Booster:
    data = lgb.Dataset(train[features].to_numpy(np.float32), train[label].to_numpy(np.float32))
    return lgb.train(params, data, num_boost_round=rounds)


def predict_lightgbm(model: lgb.Booster, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df[features].to_numpy(np.float32))
