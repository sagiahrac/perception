import random

import polars as pl

EWM_MODE = {"alpha": 0.01}


def add_bt_price_trend(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("price")
        .diff()
        .sign()
        .fill_null(random.choice([-1, 1]))
        .replace({0: None})
        .fill_null(strategy="forward")
        .alias("price_diff")
    )
    return df


def add_pbuy_psell_ewm(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [
            ((pl.col("price_diff") + 1) // 2).ewm_mean(span=1000).alias("pbuy_ewm"),
            ((pl.col("price_diff") * (-1) + 1) // 2)
            .ewm_mean(span=1000)
            .alias("psell_ewm"),
        ]
    )
    return df


def add_log_dollars_exchanged(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (pl.col("price_diff") * (pl.col("size") * pl.col("price")).log()).alias(
            "log_dollars_exchanged"
        )
    )
    return df


def add_log_dollars_exchanged_ewm(df: pl.DataFrame) -> pl.DataFrame:
    pos_idx = (df["price_diff"] == 1).arg_true()
    neg_idx = (df["price_diff"] == -1).arg_true()
    df_pos = df[pos_idx, :]
    df_neg = df[neg_idx, :]

    df_pos = (
        df_pos.with_columns(
            pl.col("log_dollars_exchanged")
            .ewm_mean(**EWM_MODE)
            .alias("log_dollars_buy_exchanged_ewm")
        )
        .with_columns(log_dollars_sell_exchanged_ewm=None)
        .with_columns(pl.col("log_dollars_sell_exchanged_ewm").cast(pl.Float64))
    )

    df_neg = (
        df_neg.with_columns(log_dollars_buy_exchanged_ewm=None)
        .with_columns(
            pl.col("log_dollars_exchanged")
            .ewm_mean(**EWM_MODE)
            .alias("log_dollars_sell_exchanged_ewm")
        )
        .with_columns(pl.col("log_dollars_buy_exchanged_ewm").cast(pl.Float64))
    )

    df = pl.concat([df_pos, df_neg]).sort("datetime")
    df = df.fill_null(strategy="forward")
    return df


def add_expected_log_dollars_exchanged(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        (
            pl.col("log_dollars_buy_exchanged_ewm").fill_null(0) * pl.col("pbuy_ewm")
            + pl.col("log_dollars_sell_exchanged_ewm").fill_null(0)
            * pl.col("psell_ewm")
        ).alias("expected_log_dollars_exchanged")
    )
    return df


def add_imbalance_thresholds(df: pl.DataFrame, ewm=False) -> pl.DataFrame:
    df = add_bt_price_trend(df)
    if ewm:
        df = add_pbuy_psell_ewm(df)
        df = add_log_dollars_exchanged(df)
        df = add_log_dollars_exchanged_ewm(df)
        df = add_expected_log_dollars_exchanged(df)
    return df


def _group_to_dollarbars(
    log_dollars_exchanged: pl.Series, expected_log_dollars: pl.Series, bar_len: int
):
    group_vals = []
    group_id = 0
    group_size = 0
    dollar_cumsum = 0
    std = log_dollars_exchanged.ewm_std(**EWM_MODE)
    for tick_dollars, expected_dollars, s in zip(
        log_dollars_exchanged, expected_log_dollars, std
    ):
        group_vals.append(group_id)
        group_size += 1
        dollar_cumsum += tick_dollars
        threshold = (abs(expected_dollars) + 0.1 * s) * bar_len
        # print(s)
        if group_size > 2 and abs(dollar_cumsum) >= threshold:
            dollar_cumsum = 0
            bar_len = 0.1 * bar_len + 0.9 * group_size
            group_id += 1
            group_size = 0
            # print(bar_len)
    return group_vals


def get_dollar_bars(df: pl.DataFrame, bar_starting_len) -> pl.DataFrame:
    """
    Generate dollar bars from a DataFrame with timestamp[datetime], price and size.

    Args:
        df (pl.DataFrame): Input DataFrame containing price and size columns.
        dollar_threshold (int): Threshold value for grouping dollar bars.

    Returns:
        pl.DataFrame: DataFrame containing dollar bars with time, open, close, high, low, and volume columns.
    """
    df = add_imbalance_thresholds(df, ewm=True)
    groups = _group_to_dollarbars(
        df["log_dollars_exchanged"],
        df["expected_log_dollars_exchanged"],
        bar_starting_len,
    )
    df = df.with_columns(pl.Series(name="group", values=groups))
    df = df.group_by("group").agg(
        [
            pl.last("datetime").alias("time"),
            pl.first("price").alias("open"),
            pl.last("price").alias("close"),
            pl.max("price").alias("high"),
            pl.min("price").alias("low"),
            pl.sum("size").alias("volume"),
            pl.count("size").alias("tick_count"),
        ]
    )
    return df.sort("time")
