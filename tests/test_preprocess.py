from perception.preprocess.dollar_bars import get_dollar_bars
from perception.preprocess.utils import add_timestamp_column, filter_datetime
from datetime import time


def test_create_dollar_bars():
    src = '/homes/sagiah/school/perception/data/trades/TSLA/2021/*.parquet'
    df = add_timestamp_column(src)
    df = get_dollar_bars(df, 5e4)
    assert df['time'].is_sorted()


def test_filter_time():
    src = '/homes/sagiah/school/perception/data/trades/JNJ/2014/2014-01-07_00-00-00.parquet'
    df = add_timestamp_column(src)
    df = filter_datetime(df, 'datetime')
    assert df['datetime'].max().time() <= time(16, 0)
    assert df['datetime'].min().time() >= time(9, 30)
