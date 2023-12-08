from datetime import time

import pytest

from perception.preprocess.dollar_bars import get_dollar_bars
from perception.preprocess.imbalance_bars import ticks_diff
from perception.preprocess.utils import read_trade_parquets, filter_datetime


@pytest.fixture
def df():
    src = "/homes/sagiah/school/perception/data/trades/JNJ/2014/2014-01-07_00-00-00.parquet"
    df = read_trade_parquets(src)
    return df


def test_create_dollar_bars():
    src = "/homes/sagiah/school/perception/data/trades/TSLA/2021/*.parquet"
    df = read_trade_parquets(src)
    df = get_dollar_bars(df, 5e4)
    assert df["time"].is_sorted()


def test_filter_time(df):
    df = filter_datetime(df, "datetime")
    assert df["datetime"].max().time() <= time(16, 0)
    assert df["datetime"].min().time() >= time(9, 30)


def test_bts(df):
    df_new = ticks_diff(df, ewm=True)

    assert df_new.shape[0] == df.shape[0]
    assert df_new.shape[1] == df.shape[1] + 2
    assert set(df_new["price_diff"].unique()) == {-1, 1}

    diff = df["price"].diff().sign()
    diff[0] = df_new[0, "price_diff"]
    for i in range(1, len(diff)):
        if diff[i] == 0:
            diff[i] = diff[i - 1]
    assert diff.equals(df_new["price_diff"].alias("price"))
