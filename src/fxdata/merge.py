"""Мэдээг 1 цагийн лаатай нэгтгэх.

Цагийн дүрэм (look-ahead-аас сэргийлэх):
  * Лаа ts = [ts, ts+1h). Таамаглал лааны ХААЛТАД (ts+1h) хийгдэнэ.
  * 12:30 UTC-д гарсан мэдээ -> 12:00 лаанд хамаарна (floor('h')).
  * Тухайн лаанд гарсан мэдээний actual/surprise-ийг зөвхөн тэр лааны хаалтаас хойш ашиглана.
  * Ирэх мэдээний ЦАГ нь урьдчилан зарлагддаг тул "дараагийн мэдээ хүртэл хэдэн цаг" шинжийг
    ашиглаж болно. Харин түүний actual-ийг хэзээ ч ашиглахгүй.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def candle_event_map(candle_index: pd.DatetimeIndex, events: pd.DataFrame) -> pd.DataFrame:
    """Мэдээ бүрийг харгалзах лаатай холбоно. Лаа байхгүй (амралт) бол дараагийн лаанд оноож тэмдэглэнэ."""
    ev = events[["event_id", "ts_utc"]].copy()
    ev["candle_ts"] = ev["ts_utc"].dt.floor("h")
    have = ev["candle_ts"].isin(candle_index)
    ev["shifted"] = False
    if (~have).any():
        pos = candle_index.searchsorted(ev.loc[~have, "candle_ts"])
        ok = pos < len(candle_index)
        idx = ev.index[~have][ok]
        ev.loc[idx, "candle_ts"] = candle_index[pos[ok]]
        ev.loc[idx, "shifted"] = True
        ev = ev.drop(index=ev.index[~have][~ok])
    return ev[["candle_ts", "event_id", "shifted"]].reset_index(drop=True)


def news_features(candles: pd.DataFrame, events: pd.DataFrame, impact: str = "high") -> pd.DataFrame:
    """Лаа бүрт (хаалтын агшинд мэдэгдэх) мэдээний шинжүүд."""
    ev = events[events["impact"] == impact].sort_values("ts_utc").copy()
    close_t = candles.index + pd.Timedelta(hours=1)
    out = pd.DataFrame(index=candles.index)

    ev["candle_ts"] = ev["ts_utc"].dt.floor("h")
    g = ev.groupby("candle_ts")
    out["ev_count"] = g.size().reindex(candles.index).fillna(0).astype(int)
    out["ev_surprise_z_sum"] = g["surprise_z"].sum(min_count=1).reindex(candles.index)
    out["ev_usd"] = ev[ev.currency == "USD"].groupby("candle_ts").size().reindex(candles.index).fillna(0).astype(int)
    out["ev_eur"] = ev[ev.currency == "EUR"].groupby("candle_ts").size().reindex(candles.index).fillna(0).astype(int)

    ct =close_t.tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    times = pd.DatetimeIndex(ev["ts_utc"]).tz_convert("UTC").tz_localize(None).to_numpy(dtype="datetime64[ns]")
    # сүүлийн мэдээ: ts_utc < close_t
    i_prev = np.searchsorted(times, ct, side="left") - 1
    # дараагийн мэдээ: ts_utc >= close_t
    i_next = np.searchsorted(times, ct, side="left")
    hrs = lambda a, b: (a - b) / np.timedelta64(1, "h")
    prev_ok = i_prev >= 0
    next_ok = i_next < len(times)
    out["hours_since_event"] = np.where(prev_ok, hrs(ct, times[np.clip(i_prev, 0, None)]), np.nan)
    out["hours_to_next_event"] = np.where(next_ok, hrs(times[np.clip(i_next, 0, len(times) - 1)], ct), np.nan) if len(times) else np.nan
    last_z = ev["surprise_z"].to_numpy()
    out["last_surprise_z"] = np.where(prev_ok, last_z[np.clip(i_prev, 0, None)], np.nan) if len(times) else np.nan
    return out
