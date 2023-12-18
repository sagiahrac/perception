from typing import List

import polars as pl

from ..preprocess.utils import group_by_date


def _group_to_dollarbars(
    df: pl.DataFrame,
):
    assert df["datetime"].is_sorted(), "df must be sorted by datetime"
    group_vals = []
    dollar_threshold = df["daily_volume"].item(0) / 1000
    group_id = 0
    dollar_cumsum = 0
    for traffic in df["price"] * df["size"]:
        group_vals.append(group_id)
        dollar_cumsum += traffic
        if dollar_cumsum >= dollar_threshold:
            group_id += 1
            dollar_cumsum = 0
    df = df.with_columns(group_id=pl.Series(group_vals))
    return df


def _get_daily_vol(df: pl.DataFrame) -> float:
    df = group_by_date(df, "datetime").agg(
        (pl.col("size") * pl.col("price")).sum().alias("daily_volume")
    )
    df = df.sort("date")
    df = df.with_columns(pl.col("daily_volume").ewm_mean(span=4).shift(1)).fill_null(
        strategy="backward"
    )
    return df


def get_dollar_bars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate dollar bars from a DataFrame with timestamp[datetime], price and size.

    Args:
        df (pl.DataFrame): Input DataFrame containing price and size columns.
        dollar_threshold (int): Threshold value for grouping dollar bars.

    Returns:
        pl.DataFrame: DataFrame containing dollar bars with time, open, close, high, low, and volume columns.
    """
    daily_vol = _get_daily_vol(df)
    df = df.with_columns(pl.col("datetime").dt.date().alias("date")).join(
        daily_vol, on="date"
    )
    df = df.group_by("date", maintain_order=True).map_groups(_group_to_dollarbars)
    df = df.group_by("date", "group_id", maintain_order=True).agg(
        [
            pl.last("datetime").alias("time"),
            pl.first("price").alias("open"),
            pl.last("price").alias("close"),
            pl.max("price").alias("high"),
            pl.min("price").alias("low"),
            pl.sum("size").alias("volume"),
            pl.std("price").alias("price_std"),
            (pl.max("price") / pl.min("price")).log().alias("log_range"),
            pl.count("price").alias("count"),
        ]
    )
    df = df.drop("date", "group_id")
    df = df.sort("time")
    df = df.with_columns(volatility=pl.col("price_std").ewm_mean(span=10))
    return df
