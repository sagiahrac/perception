from perception.features.preprocess import add_timestamp_column, get_dollar_bars


def test_create_dollar_bars():
    src = '/homes/sagiah/school/perception/data/trades/TSLA/2021/*.parquet'
    df = add_timestamp_column(src)
    df = get_dollar_bars(df, 5e4)
    assert df['time'].is_sorted()
