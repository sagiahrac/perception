import polars as pl
from glob import glob


def add_timestamp_column(parquets_path: str, dst=None):
    """
    Adds a timestamp column to the DataFrame obtained from parquet files.

    Args:
        parquets_path (Union[str, List[str]]): The path to the parquet file(s).
        dst (str, optional): The destination path to write the resulting DataFrame as a parquet file. Defaults to None.
        sort_idx (bool, optional): Whether to sort the DataFrame by the timestamp column. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the added timestamp column.

    """
    dfs = [pl.read_parquet(path, columns=['sip_timestamp', 'price', 'size']) for path in glob(parquets_path)]
    dfs = [df.cast({"sip_timestamp": pl.Int64, "price": pl.Float64, "size": pl.Float64}) for df in dfs]
    df = pl.concat(dfs)

    dt_type = pl.Datetime("ns", "America/New_York")
    df = df.with_columns(pl.col("sip_timestamp").cast(dt_type).alias("datetime"))
    df = df.sort("datetime")
    if dst is not None:
        df.write_parquet(dst)
    return df


def filter_datetime(df: pl.DataFrame, dt_colname: str):
    """
    Filter the DataFrame based on a datetime column between 09:30 to 16:00.

    Args:
        df (pl.DataFrame): The DataFrame to filter.
        dt_colname (str): The name of the datetime column.

    Returns:
        pl.DataFrame: The filtered DataFrame.
    """
    filtered_exp = pl.col(dt_colname).dt.replace_time_zone(None).cast(pl.Time).is_between(pl.time(9, 30), pl.time(16, 0))
    return df.filter(filtered_exp)
