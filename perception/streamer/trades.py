from polygon import RESTClient
import threading
from datetime import datetime, timedelta
import pytz
import pandas as pd
from pathlib import Path
import queue
from time import sleep
import os


def sip2datetime(sip_timestamp):
    return datetime.fromtimestamp(
        sip_timestamp / 1e9, tz=pytz.timezone("America/New_York")
    )


def datetime2sip(dt: datetime):
    return int(pytz.timezone("America/New_York").localize(dt).timestamp() * 1e9)


def check4trades(ticker, day: datetime, online=True):
    if day.weekday() > 4:
        return False
    if online:
        start = datetime2sip(day)
        end = datetime2sip(day + timedelta(days=1))
        client = RESTClient()
        trades = client.list_trades(
            ticker=ticker, timestamp_gte=start, timestamp_lt=end
        )
        try:
            next(trades)
        except StopIteration:
            return False
    return True


def get_trades(ticker, start, end):
    client = RESTClient()
    trades = client.list_trades(ticker=ticker, timestamp_gte=start, timestamp_lt=end)
    trades_list = [vars(trade) for trade in trades]
    return trades_list


def trades_list_to_parquet(trades_list, dst):
    if len(trades_list) > 0:
        pd.DataFrame.from_records(trades_list).to_parquet(dst, index=False)
        print(f"{dst} saved,\t{len(trades_list)} trades")
    else:
        print(f"{dst} skipped,\t0 trades")


def write_trades(ticker, start_dt: datetime, delta: timedelta, datadir):
    print(f'Start {start_dt.strftime("%Y-%m-%d")}')
    start = datetime2sip(start_dt)
    end = datetime2sip(start_dt + delta)
    dst = Path(
        f'{datadir}/{ticker}/{start_dt.strftime("%Y")}/{start_dt.strftime("%Y-%m-%d_%H-%M-%S")}.parquet'
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        message = f"{dst} skipped,\talready exists"
        with open("skipped.txt", "a") as f:
            f.write(message + "\n")
    else:
        trades_list = get_trades(ticker, start, end)
        trades_list_to_parquet(trades_list, str(dst))


def get_month_threads(ticker, year, month, datadir):
    day = datetime(year, month, 1)
    delta = timedelta(days=1)
    args4queue = []
    threads = []
    while day.month == month:
        if check4trades(ticker, day, False):
            args = (ticker, day, delta, datadir)
            args4queue.append(args)
            thread = threading.Thread(target=write_trades, args=args)
            threads.append(thread)
        day += timedelta(days=1)
    return threads, args4queue


def write_month(ticker, year, month, datadir):
    print(f"Start {year}-{month}")
    threads, _ = get_month_threads(ticker, year, month, datadir)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def write_months(ticker, year, months, datadir):
    threads = []
    for month in months:
        thread = threading.Thread(
            target=write_month, args=(ticker, year, month, datadir)
        )
        threads.append(thread)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def start_thread(q):
    while q.qsize() > 0:
        try:
            args = q.get(timeout=1)
        except queue.Empty:
            print("Queue is empty")
            break
        write_trades(*args)


# N_THREADS = 100
# if __name__ == "__main__":
#     datadir = "/homes/sagiah/school/perception/tickdata"
#     q = queue.Queue()
#     for ticker in ["TSLA", "GE", "IBM", "PG", "JPM"]:
#         print(f"Getting {ticker} trades")
#         for year in range(2014, 2023):
#             for month in range(1, 13):
#                 threads, threads_args = get_month_threads(ticker, year, month, datadir)
#                 for args in threads_args:
#                     q.put(args)
#             print(f"Got {q.qsize()} tasks for {ticker} {year}")

#     threads = []
#     for i in range(N_THREADS):
#         t = threading.Thread(target=start_thread, args=(q,))
#         threads.append(t)

#     for t in threads:
#         t.start()

#     for t in threads:
#         t.join()
