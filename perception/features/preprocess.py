from typing import List, Union
import polars as pl


def add_timestamp_column(parquets_path: Union[str, List[str]], dst=None):
    """
    Adds a timestamp column to the DataFrame obtained from parquet files.

    Args:
        parquets_path (Union[str, List[str]]): The path(s) to the parquet file(s).
        dst (str, optional): The destination path to write the resulting DataFrame as a parquet file. Defaults to None.
        sort_idx (bool, optional): Whether to sort the DataFrame by the timestamp column. Defaults to True.

    Returns:
        DataFrame: The DataFrame with the added timestamp column.

    """

    df = pl.scan_parquet(parquets_path)
    dt_type = pl.Datetime("ns", "America/New_York")
    df = df.with_columns(pl.col("sip_timestamp").cast(dt_type).alias("datetime"))
    df = df.sort("datetime").collect()
    if dst is not None:
        df.write_parquet(dst)
    return df


def _group_to_dollarbars(
    prices: pl.Series, vols: pl.Series, dollar_threshold: int
) -> List[int]:
    group_vals = []
    group_id = 0
    dollar_cumsum = 0
    for traffic in (prices * vols):
        group_vals.append(group_id)
        dollar_cumsum += traffic
        if dollar_cumsum >= dollar_threshold:
            group_id += 1
            dollar_cumsum = 0
    return group_vals


def get_dollar_bars(df: pl.DataFrame, dollar_threshold: int) -> pl.DataFrame:
    """
    Generate dollar bars from a DataFrame with timestamp[datetime], price and size.

    Args:
        df (pl.DataFrame): Input DataFrame containing price and size columns.
        dollar_threshold (int): Threshold value for grouping dollar bars.

    Returns:
        pl.DataFrame: DataFrame containing dollar bars with time, open, close, high, low, and volume columns.
    """

    groups = _group_to_dollarbars(df["price"], df["size"], dollar_threshold)
    df = df.with_columns(pl.Series(name="group", values=groups))
    df = df.group_by("group").agg(
        [
            pl.last("datetime").alias("time"),
            pl.first("price").alias("open"),
            pl.last("price").alias("close"),
            pl.max("price").alias("high"),
            pl.min("price").alias("low"),
            pl.sum("size").alias("volume"),
        ]
    )
    return df.sort("time")
