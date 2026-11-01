"""SQLite өгөгдлийн сан үүсгэх, ачаалах."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    con.execute("INSERT OR IGNORE INTO instruments(id, symbol, pip_size) VALUES (1, 'EURUSD', 0.0001)")
    return con


def _iso(s: pd.Series | pd.Index) -> list[str]:
    return [t.isoformat() for t in pd.DatetimeIndex(s).tz_convert("UTC")]


def write_candles(con, df: pd.DataFrame, source: str, timeframe="H1", unclean: pd.DatetimeIndex | None = None):
    d = df[~df.index.duplicated()]
    flag = d.index.isin(unclean) if unclean is not None else [False] * len(d)
    rows = [
        (1, timeframe, ts, source, "BID", o, h, l, c, None if pd.isna(v) else float(v), 0 if bad else 1)
        for ts, o, h, l, c, v, bad in zip(_iso(d.index), d.open, d.high, d.low, d.close, d.volume, flag)
    ]
    con.executemany(
        "INSERT OR REPLACE INTO candles(instrument_id,timeframe,ts_utc,source,price_side,open,high,low,close,volume,is_clean)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def write_issues(con, issues: pd.DataFrame):
    if issues.empty:
        return 0
    res = issues["issue"].map({"duplicate": "dropped", "ohlc_invalid": "dropped", "flat_bar": "dropped"}).fillna("kept")
    rows = list(zip(issues["source"], _iso(issues["ts"]), issues["issue"], issues["detail"].astype(str), res))
    con.executemany("INSERT INTO data_quality_issues(source,ts_utc,issue,detail,resolution) VALUES (?,?,?,?,?)", rows)
    con.commit()
    return len(rows)


def write_events(con, ev: pd.DataFrame, source="forexfactory") -> pd.DataFrame:
    """Мэдээг бичээд event_id-тай буцаана."""
    nz = lambda x: None if pd.isna(x) else float(x)
    rows = [
        (ts, r.currency, r.impact, r.title, nz(r.actual), nz(r.forecast), nz(r.previous),
         nz(r.surprise), nz(r.surprise_z), r.raw_time_text, source)
        for ts, r in zip(_iso(ev["ts_utc"]), ev.itertuples())
    ]
    con.executemany(
        "INSERT OR IGNORE INTO econ_events(ts_utc,currency,impact,title,actual,forecast,previous,surprise,surprise_z,raw_time_text,source)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    ids = pd.read_sql("SELECT event_id, ts_utc, currency, title FROM econ_events", con)
    ids["ts_utc"] = pd.to_datetime(ids["ts_utc"], utc=True)
    return ev.merge(ids, on=["ts_utc", "currency", "title"], how="left")


def write_candle_events(con, cmap: pd.DataFrame):
    rows = list(zip(_iso(cmap["candle_ts"]), cmap["event_id"].astype(int), cmap["shifted"].astype(int)))
    con.executemany("INSERT OR IGNORE INTO candle_events(candle_ts_utc,event_id,shifted) VALUES (?,?,?)", rows)
    con.commit()
    return len(rows)
