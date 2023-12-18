from datetime import datetime, timedelta
from glob import glob

import polars as pl


def read_trade_parquets(parquets_path: str, dst=None, filter_dt=True):
    """
    Adds a timestamp column to the DataFrame obtained from parquet files.

    Args:
        parquets_path (Union[str, List[str]]): The path to the parquet file(s).
        dst (str, optional): The destination path to write the resulting DataFrame as a parquet file. Defaults to None.
        sort_idx (bool, optional): Whether to sort the DataFrame by the timestamp column. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the added timestamp column.

    """
    dfs = [
        pl.read_parquet(path, columns=["sip_timestamp", "price", "size"])
        for path in glob(parquets_path)
    ]
    dfs = [
        df.cast({"sip_timestamp": pl.Int64, "price": pl.Float64, "size": pl.Float64})
        for df in dfs
    ]
    df = pl.concat(dfs)

    dt_type = pl.Datetime("ns", "America/New_York")
    df = df.with_columns(pl.col("sip_timestamp").cast(dt_type).alias("datetime"))
    df = df.drop("sip_timestamp")
    df = df.sort("datetime")

    if filter_dt:
        df = filter_datetime(df, "datetime")
    if dst is not None:
        df.write_parquet(dst)
    return df


def filter_datetime(df: pl.DataFrame, dt_colname: str, start=(9, 30), end=(16, 0)):
    """
    Filter the DataFrame based on a datetime column between 09:30 to 16:00.

    Args:
        df (pl.DataFrame): The DataFrame to filter.
        dt_colname (str): The name of the datetime column.

    Returns:
        pl.DataFrame: The filtered DataFrame.
    """
    filtered_exp = (
        pl.col(dt_colname)
        .dt.replace_time_zone(None)
        .cast(pl.Time)
        .is_between(pl.time(*start), pl.time(*end))
    )
    return df.filter(filtered_exp)


def get_time_barriers(step: timedelta, start=(9, 30), stop=(16, 0)):
    h0, m0 = start
    h1, m1 = stop

    start = datetime.now().replace(hour=h0, minute=m0, second=0, microsecond=0)
    stop = datetime.now().replace(hour=h1, minute=m1, second=0, microsecond=0)

    current = start
    barriers = []
    while current < stop:
        barriers.append(current)
        current += step
    barriers.append(stop)

    assert len(barriers) > 2, "[WARNING] Not enough barriers"
    assert (
        barriers[-1] - barriers[-2] == step
    ), "[WARNING] Last interval is not equal to step"
    return [b.time() for b in barriers]


def split_to_time_intervals(
    df: pl.DataFrame, dt_colname: str, step: timedelta, start=(9, 30), stop=(16, 0)
):
    """
    Split the DataFrame into time intervals.

    Args:
        df (pl.DataFrame): The DataFrame to split.
        dt_colname (str): The name of the datetime column.
        step (timedelta): The time interval.
        start (tuple, optional): The start time. Defaults to (9,30).
        stop (tuple, optional): The stop time. Defaults to (16,0).

    Returns:
        List[pl.DataFrame]: The list of DataFrames split by time intervals.
    """
    barriers = get_time_barriers(step, start, stop)
    dfs = []
    for i in range(len(barriers) - 1):
        filtered_exp = (
            pl.col(dt_colname)
            .dt.replace_time_zone(None)
            .cast(pl.Time)
            .is_between(barriers[i], barriers[i + 1])
        )
        dfs.append(df.filter(filtered_exp))
    return dfs

def group_by_date(df: pl.DataFrame, dt_colname: str):
    """
    Group the DataFrame by day.

    Args:
        df (pl.DataFrame): The DataFrame to group.
        dt_colname (str): The name of the datetime column.

    Returns:
        List[pl.DataFrame]: The list of DataFrames grouped by day.
    """
    df = df.with_columns(pl.col(dt_colname).dt.date().alias("date"))
    return df.group_by("date")

def calc_stds_for_windows(bars: pl.DataFrame, minutes: int):
    dfs = split_to_time_intervals(bars, 'time', timedelta(minutes=minutes))
    for i, df in enumerate(dfs):
        df = df.with_columns(date=pl.col('time').dt.date().alias('date'))
        stds = df.group_by('date').agg(pl.col('close').std().alias('std'))
        df = df.join(stds, on='date').sort('date')
        dfs[i] = df

    df = pl.concat(dfs).sort('time')
    return df