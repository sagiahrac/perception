import os
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree

import pandas as pd

from perception.streamer.trades import (check4trades, datetime2sip, get_trades,
                                        sip2datetime, write_trades)


def test_polygon_online():
    assert os.environ.get("POLYGON_API_KEY") is not None


def test_check_tradeday():
    ticker = "ADBE"
    assert not check4trades(ticker, datetime(2020, 1, 1))
    assert check4trades(ticker, datetime(2020, 1, 2))
    assert check4trades(ticker, datetime(2020, 1, 2))
    assert check4trades(ticker, datetime(2020, 1, 3))
    assert not check4trades(ticker, datetime(2020, 1, 4))
    assert not check4trades(ticker, datetime(2020, 1, 5))


def test_sip2datetime():
    dt = sip2datetime(datetime2sip(datetime(2020, 1, 1)))
    assert dt.strftime("%Y-%m-%d %H:%M:%S") == "2020-01-01 00:00:00"


def test_get_trades():
    dt = datetime(2020, 1, 2, 9, 30, 0)
    start = datetime2sip(dt)
    end = datetime2sip(dt + timedelta(seconds=1))
    trades = get_trades("NFLX", start, end)
    assert len(trades) > 0


def test_write_trades():
    datadir = "/tmp/finance_tests"
    Path(datadir).mkdir()
    dt = datetime(2020, 1, 2, 9, 30, 0)
    delta = timedelta(seconds=1)
    write_trades("NFLX", dt, delta, datadir)

    df = pd.read_parquet(f"{datadir}")
    assert df.shape[0] > 0
    assert "size" in df.columns
    rmtree(datadir)
