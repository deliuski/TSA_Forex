"""5 минутын (M5) өгөгдөл дээрх ирэх 1 цагийн хүрээний таамаглал: олон timeframe-ийн шинжүүд.

Олон timeframe-ийн зарчим: бага timeframe (M5) дээр таамаглаж, өндөр timeframe (H1, H4)-ийн
нөхцөлийг шинж болгон нэмнэ. Look-ahead урхи: H1 лаа (жишээ нь 10:00) 11:00-д л ХААГДАНА.
Тиймээс H1/H4 шинжийг ХААЛТЫН цагаар нь индекслээд, M5 лааны хаалтын цагтай "өмнөх хамгийн
сүүлийн" (merge_asof backward) аргаар тааруулна.

Бүх цаг UTC. M5 лааны цаг = нээлтийн цаг; таамаглал лааны хаалтад (ts + 5 мин) хийгдэнэ.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from fxdata import prices
from .features import atr, base_features, ict_features

PIP = 1e-4
BAR = pd.Timedelta(minutes=5)


def load_m1_all(folder: Path) -> pd.DataFrame:
    """HistData M1 (folder/*.zip) + Dukascopy M1 (folder/../dukascopy_m1/*.csv, HistData дууссаны дараах хэсэг).
    Хоёулаа BID. HistData 2025 онд дуусдаг тул 2026 оныг Dukascopy-оос нөхнө (scripts/download_m1_dukascopy.py)."""
    m1 = prices.load_histdata_m1(sorted(glob.glob(str(Path(folder) / "*.zip"))))
    m1 = m1[~m1.index.duplicated()]
    dm = sorted((Path(folder).parent / "dukascopy_m1").glob("*.csv"))
    if dm:
        d = prices.load_dukascopy_csv(dm)
        d = d[~d.index.duplicated()]
        d = d[d.index > m1.index.max()]
        m1 = pd.concat([m1, d[m1.columns]]).sort_index()
    return m1


def load_m5_cached(root: Path) -> pd.DataFrame:
    """data/processed/eurusd_m5.parquet-ийг ашиглана; түүхий M1 файл кэшээс шинэ бол дахин үүсгэнэ."""
    cache = Path(root) / "data/processed/eurusd_m5.parquet"
    raw = list((Path(root) / "data/raw/histdata").glob("*.zip")) + list((Path(root) / "data/raw/dukascopy_m1").glob("*.csv"))
    if cache.exists() and (not raw or cache.stat().st_mtime >= max(f.stat().st_mtime for f in raw)):
        return pd.read_parquet(cache)
    m5 = load_m5_from_histdata(Path(root) / "data/raw/histdata")
    cache.parent.mkdir(parents=True, exist_ok=True)
    m5.to_parquet(cache)
    return m5


def load_m5_from_histdata(folder: Path) -> pd.DataFrame:
    m1 = load_m1_all(folder)
    m5 = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "count"}).dropna(subset=["open"])
    return m5.rename(columns={"volume": "n_m1"})


def resample(m5: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = m5.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "n_m1": "sum"}).dropna(subset=["open"])
    return out.rename(columns={"n_m1": "volume"})


def future_range_bars(m5: pd.DataFrame, n: int = 12) -> pd.Series:
    """Дараагийн n лааны хүрээ (пип). Лаа дутуу эсвэл амралтын өдөр дамнавал NaN."""
    hi = m5["high"][::-1].rolling(n, min_periods=n).max()[::-1].shift(-1)
    lo = m5["low"][::-1].rolling(n, min_periods=n).min()[::-1].shift(-1)
    r = (hi - lo) / PIP
    span = pd.Series(m5.index).shift(-n).to_numpy() - m5.index.to_numpy()
    r[span != np.timedelta64(n * 5, "m")] = np.nan
    return r


def _log_range(m5: pd.DataFrame, n: int) -> pd.Series:
    return np.log(((m5["high"].rolling(n).max() - m5["low"].rolling(n).min()) / PIP).clip(lower=0.5))


def m5_base(m5: pd.DataFrame) -> pd.DataFrame:
    """M5 суурь + өдөр доторх HAR шинжүүд."""
    c, a = m5["close"], atr(m5)
    f = pd.DataFrame(index=m5.index)
    for k in (1, 3, 6, 12, 48):
        f[f"m5_ret{k}"] = (c - c.shift(k)) / a
    f["m5_atr_ratio"] = a / atr(m5, 100)
    f["m5_bar_atr"] = (m5["high"] - m5["low"]) / a
    f["m5_completeness"] = m5["n_m1"].rolling(12).mean() / 5
    close_t = m5.index + BAR
    mod = close_t.hour * 60 + close_t.minute
    f["tod_sin"] = np.sin(2 * np.pi * mod / 1440)
    f["tod_cos"] = np.cos(2 * np.pi * mod / 1440)
    f["dow"] = close_t.dayofweek
    # Өдөр доторх HAR (log): сүүлийн 1ц, 4ц, 24ц, 5 өдрийн дундаж
    f["har_1h"] = _log_range(m5, 12)
    f["har_4h"] = _log_range(m5, 48)
    f["har_1d"] = _log_range(m5, 288)
    f["har_5d"] = sum(f["har_1d"].shift(288 * k) for k in range(5)) / 5
    return f


def seasonal_baseline(m5: pd.DataFrame, y_log: pd.Series, days: int = 20) -> pd.Series:
    """Өдрийн ижил цагийн (5 минутын слот) сүүлийн `days` өдрийн ирээдүйн хүрээний медиан (log).
    shift(1): өчигдрийн ижил слотын 1 цагийн хүрээ аль хэдийн бүрэн мэдэгдсэн байна."""
    slot = (m5.index + BAR).hour * 12 + (m5.index + BAR).minute // 5
    return y_log.groupby(slot).transform(lambda s: s.rolling(days, min_periods=5).median().shift(1))


def htf_context(m5: pd.DataFrame) -> pd.DataFrame:
    """H1, H4 шинжүүдийг M5 лааны хаалтын цагтай look-ahead-гүй тааруулна."""
    out = pd.DataFrame(index=m5.index)
    m5_close = pd.DataFrame({"t": m5.index + BAR}, index=m5.index)
    specs = {
        "h1": ("1h", ["ret_6", "ret_24", "atr_ratio_14_100", "range24_atr", "vol_ratio_24_240"],
               ["structure", "premium_discount", "dist_swing_high_atr", "dist_swing_low_atr", "htf_pos_day"]),
        "h4": ("4h", ["ret_6", "atr_ratio_14_100"], ["structure", "premium_discount", "htf_pos_week"]),
    }
    for name, (rule, bcols, icols) in specs.items():
        tf = resample(m5, rule)
        feats = base_features(tf)[bcols].join(ict_features(tf)[icols])
        feats.columns = [f"{name}_{c}" for c in feats.columns]
        feats["t"] = tf.index + pd.Timedelta(rule)            # ХААЛТЫН цаг
        merged = pd.merge_asof(m5_close.sort_values("t"), feats.sort_values("t"), on="t", direction="backward")
        merged.index = m5_close.sort_values("t").index
        out = out.join(merged.drop(columns="t"))
    return out


def news_minutes(m5: pd.DataFrame, events: pd.DataFrame, impact: str = "high", z_clip: float = 5.0) -> pd.DataFrame:
    """Мэдээний шинж минутаар: дараагийн/сүүлийн мэдээ хүртэлх минут, ирэх 60 минутын мэдээний тоо."""
    ev = events[events["impact"] == impact].sort_values("ts_utc")
    times = pd.DatetimeIndex(ev["ts_utc"]).tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
    ct = (m5.index + BAR).tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
    f = pd.DataFrame(index=m5.index)
    mins = lambda a, b: (a - b) / np.timedelta64(1, "m")  # noqa: E731
    # Хаалтын агшинтай яг давхцаж гарсан мэдээг ИРЭЭДҮЙД тооцно (лааны үнэ түүний хариуг агуулаагүй)
    i_next = np.searchsorted(times, ct, side="left")
    i_prev = i_next - 1
    ok_n, ok_p = i_next < len(times), i_prev >= 0
    f["min_to_next_event"] = np.where(ok_n, mins(times[np.clip(i_next, 0, len(times) - 1)], ct), np.nan)
    f["min_since_event"] = np.where(ok_p, mins(ct, times[np.clip(i_prev, 0, None)]), np.nan)
    f["min_to_next_event"] = f["min_to_next_event"].clip(upper=7 * 1440)
    f["min_since_event"] = f["min_since_event"].clip(upper=7 * 1440)
    f["events_next_60m"] = np.searchsorted(times, ct + np.timedelta64(60, "m"), side="left") - i_next
    z = ev["surprise_z"].clip(-z_clip, z_clip).to_numpy()
    f["last_surprise_z_abs"] = np.where(ok_p, np.abs(z[np.clip(i_prev, 0, None)]), np.nan)
    return f
