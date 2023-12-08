import random

import polars as pl


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
            .ewm_mean(span=500)
            .alias("log_dollars_buy_exchanged_ewm")
        )
        .with_columns(log_dollars_sell_exchanged_ewm=None)
        .with_columns(pl.col("log_dollars_sell_exchanged_ewm").cast(pl.Float64))
    )

    df_neg = (
        df_neg.with_columns(log_dollars_buy_exchanged_ewm=None)
        .with_columns(
            pl.col("log_dollars_exchanged")
            .ewm_mean(span=500)
            .alias("log_dollars_sell_exchanged_ewm")
        )
        .with_columns(pl.col("log_dollars_buy_exchanged_ewm").cast(pl.Float64))
    )

    df = pl.concat([df_pos, df_neg]).sort("datetime")
    df = df.drop("log_dollars_exchanged")
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
