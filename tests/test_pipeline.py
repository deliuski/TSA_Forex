"""Синтетик (зохиомол) өгөгдөл дээрх тест. Бодит өгөгдөл биш — зөвхөн кодын логикийг шалгана."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fxdata import calendar, db, merge, prices, quality  # noqa: E402


def synth_h1(start="2024-03-01", end="2024-03-29", seed=0):
    idx = quality.expected_trading_hours(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))
    rng = np.random.default_rng(seed)
    close = 1.08 + np.cumsum(rng.normal(0, 0.0008, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + rng.uniform(0.0002, 0.0010, len(idx))
    lo = np.minimum(open_, close) - rng.uniform(0.0002, 0.0010, len(idx))
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": close,
                         "volume": rng.uniform(1000, 5000, len(idx))}, index=pd.DatetimeIndex(idx, name="ts"))


# ---------- Цагийн бүс ----------
def test_trading_week_follows_ny_dst():
    idx = quality.expected_trading_hours(pd.Timestamp("2024-03-01", tz="UTC"), pd.Timestamp("2024-03-20", tz="UTC"))
    sundays = [t for t in idx if t.dayofweek == 6]
    first_by_day = pd.Series(sundays).groupby(pd.Series(sundays).dt.date).min()
    # 3/3 (өвлийн цаг) Ням 22:00 UTC, 3/10-аас (зуны цаг) 21:00 UTC-д нээгдэнэ
    assert first_by_day.iloc[0].hour == 22
    assert first_by_day.iloc[-1].hour == 21


def test_histdata_follows_new_york_dst(tmp_path):
    # Бодит файлуудыг Dukascopy-тай тулгахад HistData нь Нью-Йоркийн зуны цагийг дагадаг нь батлагдсан.
    f = tmp_path / "h.csv"
    f.write_text("20240715 080000;1.09;1.091;1.089;1.0905;0\n20240715 080100;1.0905;1.0906;1.0904;1.0905;0\n"
                 "20240115 080000;1.09;1.091;1.089;1.0905;0\n")
    m1 = prices.load_histdata_m1(f)
    assert m1.index[0] == pd.Timestamp("2024-01-15 13:00", tz="UTC")   # өвөл: UTC-5
    assert m1.index[1] == pd.Timestamp("2024-07-15 12:00", tz="UTC")   # зун: UTC-4
    h1 = prices.resample_to_h1(m1)
    assert h1.loc[pd.Timestamp("2024-07-15 12:00", tz="UTC"), "high"] == 1.091


def test_calendar_tehran_dst_change(tmp_path):
    # NFP хоёулаа 12:30 UTC. Тегеран 2015 онд зуны цагтай (UTC+4:30), 2023 онд байхгүй (UTC+3:30).
    f = tmp_path / "cal.csv"
    pd.DataFrame({
        "DateTime": ["2015-07-02 17:00:00", "2023-07-07 16:00:00", "2023-07-08 All Day"],
        "Currency": ["USD", "USD", "USD"],
        "Impact": ["High Impact Expected"] * 3,
        "Event": ["Non-Farm Employment Change", "Non-Farm Employment Change", "Bank Holiday"],
        "Actual": ["223K", "209K", ""], "Forecast": ["230K", "225K", ""], "Previous": ["254K", "306K", ""],
    }).to_csv(f, index=False)
    ev, log = calendar.load_calendar(f, "Asia/Tehran")
    assert list(ev.ts_utc.dt.strftime("%H:%M")) == ["12:30", "12:30"]
    assert log["time_textual"] == 1 and log["time_unknown_dropped"] == 1
    assert ev.surprise.tolist() == [-7000.0, -16000.0]


def _cal(tmp_path, rows):
    f = tmp_path / "cal.csv"
    pd.DataFrame(rows, columns=["DateTime", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"]).to_csv(f, index=False)
    return calendar.load_calendar(f, "Asia/Tehran")


def test_calendar_ignores_wrong_file_offset_after_iran_dst_abolished(tmp_path):
    # 2024 онд Иранд зуны цаг байхгүй (+03:30) ч файлд +04:30 гэж бичсэн. 16:00 Тегеран = 12:30 UTC.
    ev, log = _cal(tmp_path, [["2024-07-11T16:00:00+04:30", "USD", "High Impact Expected", "Unemployment Claims", "222K", "236K", "238K"]])
    assert ev.ts_utc.iloc[0] == pd.Timestamp("2024-07-11 12:30", tz="UTC")
    assert log["file_offset_disagrees_with_tzdata"] == 1


def test_calendar_imputes_placeholder_times(tmp_path):
    hi = "High Impact Expected"
    ev, log = _cal(tmp_path, [
        ["2024-04-05T00:00:00+04:30", "USD", hi, "Non-Farm Employment Change", "303K", "212K", "270K"],  # зун -> 12:30 UTC
        ["2024-02-02T00:00:00+03:30", "USD", hi, "Non-Farm Employment Change", "353K", "187K", "333K"],  # өвөл -> 13:30 UTC
        ["2022-06-09T00:00:00+04:30", "EUR", hi, "Main Refinancing Rate", "0.00%", "0.00%", "0.00%"],     # 13:45 CEST
        ["2023-06-15T00:00:00+04:30", "EUR", hi, "Main Refinancing Rate", "4.00%", "4.00%", "3.75%"],     # 14:15 CEST
        ["2024-04-05T00:00:00+04:30", "USD", hi, "FOMC Member Kugler Speaks", "", "", ""],                 # тодорхойгүй
    ])
    got = dict(zip(ev.title + " " + ev.ts_utc.dt.strftime("%Y"), ev.ts_utc.dt.strftime("%m-%d %H:%M")))
    assert got["Non-Farm Employment Change 2024"] in ("04-05 12:30", "02-02 13:30")
    assert sorted(ev.loc[ev.title == "Non-Farm Employment Change", "ts_utc"].dt.strftime("%m-%d %H:%M")) == ["02-02 13:30", "04-05 12:30"]
    assert got["Main Refinancing Rate 2022"] == "06-09 11:45"
    assert got["Main Refinancing Rate 2023"] == "06-15 12:15"
    assert "FOMC Member Kugler Speaks 2024" not in got and log["time_unknown_dropped"] == 1
    assert set(ev.time_source) == {"schedule"}


@pytest.mark.parametrize("s,v", [("175K", 175000), ("2.4%", 2.4), ("-0.1%", -0.1), ("1.2B", 1.2e9),
                                 ("5.31|2.6", 5.31), ("", np.nan), ("<0.50%", 0.5), ("1,234K", 1234000)])
def test_parse_value(s, v):
    r = calendar.parse_value(s)
    assert (np.isnan(r) and np.isnan(v)) or r == pytest.approx(v)


# ---------- Чанарын шалгалт ----------
def test_quality_detects_injected_errors():
    df = synth_h1()
    gap_ts = df.index[100]
    spike_ts = df.index[200]
    bad_ts = df.index[300]
    df.loc[spike_ts, "high"] += 0.0200            # 200 пипийн "spike"
    df.loc[bad_ts, "high"] = df.loc[bad_ts, "low"] - 0.0001  # OHLC зөрчил
    df = df.drop(index=gap_ts)
    df = pd.concat([df, df.iloc[[50]]]).sort_index()  # давхардал
    iss = quality.check_prices(df, source="t")
    got = {i: set(iss.loc[iss.issue == i, "ts"]) for i in iss.issue.unique()}
    assert gap_ts in got["missing_hour"]
    assert spike_ts in got["spike"]
    assert bad_ts in got["ohlc_invalid"]
    assert df.index[50] in got["duplicate"]
    clean = quality.clean_prices(df, iss)
    assert bad_ts not in clean.index and clean.index.is_unique


def test_compare_sources_detects_one_hour_shift():
    a = synth_h1()
    b = a.copy()
    b.index = b.index - pd.Timedelta(hours=1)  # цагийн бүсийн алдааг дуурайлгав
    r = quality.compare_sources(a, b)
    assert r["close_mean_pips_shift+1h"] < 0.01 < r["close_mean_pips"]


# ---------- Нэгтгэл ба look-ahead ----------
def _events():
    return pd.DataFrame({
        "event_id": [1, 2],
        "ts_utc": pd.to_datetime(["2024-03-08 13:30", "2024-03-09 10:00"], utc=True),  # 2 дахь нь Бямба
        "currency": ["USD", "EUR"], "impact": ["high", "high"], "title": ["NFP", "X"],
        "surprise_z": [1.5, -0.5],
    })


def test_event_maps_to_floor_hour_and_weekend_shift():
    c = synth_h1()
    m = merge.candle_event_map(c.index, _events())
    assert m.loc[m.event_id == 1, "candle_ts"].iloc[0] == pd.Timestamp("2024-03-08 13:00", tz="UTC")
    row = m[m.event_id == 2].iloc[0]
    assert row.shifted and row.candle_ts.dayofweek == 6  # Ням оройн нээлтийн лаа


def test_news_features_no_lookahead():
    c = synth_h1()
    f = merge.news_features(c, _events())
    before = f.loc[pd.Timestamp("2024-03-08 12:00", tz="UTC")]  # хаалт 13:00 < 13:30
    at = f.loc[pd.Timestamp("2024-03-08 13:00", tz="UTC")]      # хаалт 14:00 > 13:30
    assert before.hours_to_next_event == pytest.approx(0.5)
    assert np.isnan(before.last_surprise_z)        # мэдээний actual-ийг урьдчилж мэдэхгүй
    assert at.ev_count == 1 and at.last_surprise_z == 1.5
    assert at.hours_since_event == pytest.approx(0.5)


def test_db_roundtrip(tmp_path):
    con = db.connect(tmp_path / "t.db")
    c = synth_h1()
    assert db.write_candles(con, c, "dukascopy") == len(c)
    ev = _events().assign(actual=1.0, forecast=0.5, previous=0.2, surprise=0.5, raw_time_text="x")
    ev = db.write_events(con, ev.drop(columns="event_id"))
    db.write_candle_events(con, merge.candle_event_map(c.index, ev))
    n = con.execute("SELECT COUNT(*) FROM candle_events ce JOIN econ_events e USING(event_id)").fetchone()[0]
    assert n == 2
