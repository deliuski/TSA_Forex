"""Ханшийн өгөгдлийн чанарын шалгалт.

Зарчим: алдааг ЧИМЭЭГҮЙ устгахгүй. Бүгдийг илрүүлж, issues хүснэгтэд бүртгэж,
тайланд тоогоор нь гаргана. Засах шийдвэр тусдаа, тайлбартай хийгдэнэ.

Шалгалтууд:
  duplicate     - нэг цаг давхардсан
  ohlc_invalid  - high < max(open,close), low > min(open,close), эсвэл үнэ <= 0
  missing_hour  - арилжааны долоо хоног дотор байх ёстой цаг алга
  flat_bar      - O=H=L=C эсвэл volume=0 (арилжаа болоогүй/дүүргэсэн лаа)
  spike         - лааны хүрээ сүүлийн 10 хоногийн медианаас 8 дахин их
  return_jump   - цагийн өгөөж робаст z-оноогоор 10-аас их
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
PIP = 1e-4  # EUR/USD


@dataclass
class QualityConfig:
    spike_k: float = 8.0
    spike_window: int = 240        # ~10 арилжааны өдөр
    jump_z: float = 10.0
    holiday_days: tuple = ((12, 25), (1, 1))  # эдгээр өдрийн цоорхойг 'holiday' гэж ангилна


def expected_trading_hours(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Форекс долоо хоног: Ням 17:00 NY-аас Баасан 17:00 NY хүртэл.
    Нью-Йоркийн зуны цагийг тооцдог тул UTC-д 21:00 эсвэл 22:00 болж солигдоно."""
    idx = pd.date_range(start.floor("h"), end.floor("h"), freq="h", tz="UTC")
    ny = idx.tz_convert(NY)
    dow, hour = ny.dayofweek, ny.hour  # Даваа=0 ... Ням=6
    closed = (
        (dow == 5)                         # Бямба
        | ((dow == 4) & (hour >= 17))      # Баасан 17:00-оос хойш
        | ((dow == 6) & (hour < 17))       # Ням 17:00 хүртэл
    )
    return idx[~closed]


def check_prices(df: pd.DataFrame, cfg: QualityConfig | None = None, source: str = "") -> pd.DataFrame:
    """Илэрсэн бүх асуудлыг [ts, issue, detail, source] хүснэгтээр буцаана."""
    cfg = cfg or QualityConfig()
    issues: list[pd.DataFrame] = []

    def add(ts, issue, detail):
        if len(ts):
            issues.append(pd.DataFrame({"ts": ts, "issue": issue, "detail": detail}))

    dup = df.index[df.index.duplicated(keep="first")]
    add(dup, "duplicate", "давхардсан цаг")
    d = df[~df.index.duplicated(keep="first")]

    bad = (
        (d["high"] < d[["open", "close"]].max(axis=1))
        | (d["low"] > d[["open", "close"]].min(axis=1))
        | (d[["open", "high", "low", "close"]] <= 0).any(axis=1)
    )
    add(d.index[bad], "ohlc_invalid", "OHLC логик зөрчил")

    if len(d):
        exp = expected_trading_hours(d.index.min(), d.index.max())
        missing = exp.difference(d.index)
        md = [(m.month, m.day) for m in missing.tz_convert(NY)]
        is_hol = np.array([x in cfg.holiday_days for x in md], dtype=bool)
        add(missing[~is_hol], "missing_hour", "арилжааны цаг алга")
        add(missing[is_hol], "missing_holiday", "баярын өдөр (хүлээгдэж буй)")

    flat = (d["high"] == d["low"]) | (d["volume"].fillna(1) == 0)
    add(d.index[flat], "flat_bar", "хөдөлгөөнгүй/0 эзлэхүүн")

    rng = d["high"] - d["low"]
    med = rng.rolling(cfg.spike_window, min_periods=24).median().shift(1)
    spike = rng > cfg.spike_k * med
    add(d.index[spike], "spike", [f"хүрээ {r / PIP:.1f} пип, медиан {m / PIP:.1f}" for r, m in zip(rng[spike], med[spike])])

    r = np.log(d["close"]).diff()
    mad = (r - r.rolling(cfg.spike_window, min_periods=24).median()).abs().rolling(cfg.spike_window, min_periods=24).median().shift(1)
    z = (r / (1.4826 * mad)).abs()
    jump = z > cfg.jump_z
    add(d.index[jump], "return_jump", [f"z={v:.1f}" for v in z[jump]])

    if not issues:
        return pd.DataFrame(columns=["ts", "issue", "detail", "source"])
    out = pd.concat(issues, ignore_index=True)
    out["source"] = source
    return out.sort_values(["ts", "issue"]).reset_index(drop=True)


def compare_sources(a: pd.DataFrame, b: pd.DataFrame, name_a="dukascopy", name_b="histdata") -> dict:
    """Хоёр эх сурвалжийн ижил цагийн лааг пипээр харьцуулна."""
    a = a[~a.index.duplicated()]
    b = b[~b.index.duplicated()]
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    res = {
        "n_a": len(a), "n_b": len(b), "n_common": len(j),
        "only_a": len(a.index.difference(b.index)), "only_b": len(b.index.difference(a.index)),
    }
    for col in ["open", "high", "low", "close"]:
        diff = (j[f"{col}_a"] - j[f"{col}_b"]).abs() / PIP
        res[f"{col}_mean_pips"] = float(diff.mean()) if len(diff) else float("nan")
        res[f"{col}_p99_pips"] = float(diff.quantile(0.99)) if len(diff) else float("nan")
        res[f"{col}_gt5pips"] = int((diff > 5).sum())
    cd = ((j["close_a"] - j["close_b"]).abs() / PIP).sort_values(ascending=False)
    res["worst_close_hours"] = [(str(t), round(v, 1)) for t, v in cd.head(10).items()]
    res["names"] = (name_a, name_b)
    # Цагийн шилжилт шалгах: b-г ±1 цагаар шилжүүлэхэд илүү таарч байвал timezone алдаа
    for shift in (-1, 1):
        bs = b.copy()
        bs.index = bs.index + pd.Timedelta(hours=shift)
        js = a.join(bs, how="inner", lsuffix="_a", rsuffix="_b")
        res[f"close_mean_pips_shift{shift:+d}h"] = float(((js["close_a"] - js["close_b"]).abs() / PIP).mean()) if len(js) else float("nan")
    return res


def summarize(issues: pd.DataFrame) -> pd.DataFrame:
    if issues.empty:
        return pd.DataFrame(columns=["issue", "count"])
    s = issues.groupby("issue").size().rename("count").reset_index()
    return s.sort_values("count", ascending=False)


def clean_prices(df: pd.DataFrame, issues: pd.DataFrame) -> pd.DataFrame:
    """Хамгийн бага хөндлөнгийн цэвэрлэгээ: давхардал, OHLC-логик зөрчилтэй, flat мөрийг хасна.
    Spike-ийг устгахгүй — мэдээний цагт бодит том хөдөлгөөн байдаг. Spike-ийг хоёр дахь
    эх сурвалжтай тулгаж, нөгөөд нь байхгүй бол л засна (тайланд тусад нь бичнэ)."""
    d = df[~df.index.duplicated(keep="first")]
    drop = issues.loc[issues["issue"].isin(["ohlc_invalid", "flat_bar"]), "ts"]
    return d.drop(index=pd.DatetimeIndex(drop).intersection(d.index))
