"""Эдийн засгийн мэдээний календарь унших, UTC болгох, утгыг тоо болгох, surprise тооцох.

Гол эрсдэл: ЦАГИЙН БҮС. ForexFactory-ийн цаг нь тухайн scrape хийсэн хэрэглэгчийн
тохиргооны бүсээр харагддаг. Жишээ нь HuggingFace дээрх 2007–2025 датасет Asia/Tehran
цагаар хадгалагдсан, Иран 2022 оноос хойш зуны цаг хэрэглэхээ больсон. Тиймээс тогтмол
offset нэмэхгүй, заавал IANA бүсийн нэрээр (zoneinfo) хөрвүүлнэ.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COLUMN_CANDIDATES = {
    "datetime": ["datetime", "date_time", "timestamp", "time_utc", "date"],
    "time": ["time"],
    "currency": ["currency", "country", "ccy"],
    "impact": ["impact", "importance"],
    "title": ["event", "title", "name"],
    "actual": ["actual"],
    "forecast": ["forecast", "consensus"],
    "previous": ["previous", "prior"],
}

_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_NUM = re.compile(r"^([<>]?)(-?\d+(?:\.\d+)?)([KMBT%]?)$")


def parse_value(s) -> float:
    """'175K' -> 175000, '2.4%' -> 2.4, '-0.1%' -> -0.1, '5.31|2.6' -> 5.31, '' -> NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().replace(",", "")
    if not s:
        return np.nan
    s = s.split("|")[0]
    m = _NUM.match(s)
    if not m:
        return np.nan
    val = float(m.group(2))
    suf = m.group(3)
    return val * _SUFFIX.get(suf, 1.0)


def normalize_impact(x) -> str:
    x = str(x).lower()
    for k in ("high", "medium", "low", "holiday"):
        if k in x:
            return k
    if x in ("3", "red"):
        return "high"
    if x in ("2", "orange"):
        return "medium"
    if x in ("1", "yellow"):
        return "low"
    return "unknown"


def _find(cols: dict, key: str):
    for c in COLUMN_CANDIDATES[key]:
        if c in cols:
            return cols[c]
    return None


def load_calendar(path, source_tz: str, currencies=("USD", "EUR")) -> tuple[pd.DataFrame, dict]:
    """CSV календарийг уншиж UTC болгоно. (events, log) буцаана.

    source_tz: CSV дахь цаг ямар бүсээр бичигдсэн ('Asia/Tehran', 'America/New_York', 'UTC' ...).
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in raw.columns}
    m = {k: _find(cols, k) for k in COLUMN_CANDIDATES}
    for need in ("datetime", "currency", "impact", "title"):
        if m[need] is None:
            raise ValueError(f"Календарьт '{need}' багана олдсонгүй. Байгаа баганууд: {list(raw.columns)}")

    log = {"rows_raw": len(raw)}
    dt_text = raw[m["datetime"]].str.strip()
    # Огноо, цаг тусдаа баганад байвал нийлүүлнэ (огнооны баганад цаг аль хэдийн байвал нийлүүлэхгүй)
    if m["time"] is not None and m["time"] != m["datetime"] and dt_text.str.contains(":").mean() < 0.5:
        dt_text = dt_text + " " + raw[m["time"]].str.strip()

    # "All Day", "Tentative", "Day 1" гэх мэт тодорхой цаггүй мөр
    time_known = ~dt_text.str.contains(r"all day|tentative|day \d", case=False, regex=True)
    txt = dt_text.where(time_known)
    # Цаг өөрөө UTC зөрүүтэй бол ("…+04:30") түүгээр нь шууд UTC болгоно — хамгийн найдвартай.
    # Зөрүүгүй бол source_tz бүсээр (зуны цагийг тооцож) хөрвүүлнэ.
    has_off = txt.str.contains(r"(?:[+-]\d{2}:?\d{2}|Z)$", regex=True, na=False)
    ts_utc = pd.Series(pd.NaT, index=txt.index, dtype="datetime64[ns, UTC]")
    if has_off.any():
        ts_utc[has_off] = pd.to_datetime(txt[has_off], errors="coerce", format="mixed", utc=True)
    naive = time_known & ~has_off
    if naive.any():
        local = pd.to_datetime(txt[naive], errors="coerce", format="mixed")
        local = local.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
        ts_utc[naive] = local.dt.tz_convert("UTC")
    log["time_with_offset"] = int(has_off.sum())

    ev = pd.DataFrame({
        "ts_utc": ts_utc,
        "currency": raw[m["currency"]].str.strip().str.upper(),
        "impact": raw[m["impact"]].map(normalize_impact),
        "title": raw[m["title"]].str.strip(),
        "actual_raw": raw[m["actual"]] if m["actual"] else "",
        "forecast_raw": raw[m["forecast"]] if m["forecast"] else "",
        "previous_raw": raw[m["previous"]] if m["previous"] else "",
        "raw_time_text": dt_text,
    })
    log["time_unknown"] = int((~time_known).sum())
    log["time_unparsed_or_dst_gap"] = int((time_known & ev["ts_utc"].isna()).sum())
    ev = ev[ev["ts_utc"].notna()]
    ev = ev[ev["currency"].isin(currencies)]
    for c in ("actual", "forecast", "previous"):
        ev[c] = ev[f"{c}_raw"].map(parse_value)
    ev["surprise"] = ev["actual"] - ev["forecast"]
    before = len(ev)
    ev = ev.drop_duplicates(subset=["ts_utc", "currency", "title"])
    log["duplicates_dropped"] = before - len(ev)
    ev = ev.sort_values("ts_utc").reset_index(drop=True)
    ev["surprise_z"] = surprise_z(ev)
    log["rows_kept"] = len(ev)
    log["high_impact"] = int((ev["impact"] == "high").sum())
    return ev, log


def surprise_z(ev: pd.DataFrame, min_periods: int = 8) -> pd.Series:
    """Мэдээ тус бүрийн surprise-ийг ӨМНӨХ гаргалтуудын std-д хуваана (ирээдүйн мэдээлэл ашиглахгүй)."""
    def _z(g: pd.Series) -> pd.Series:
        sd = g.expanding(min_periods=min_periods).std().shift(1)
        return g / sd.replace(0, np.nan)
    return ev.groupby(["currency", "title"], group_keys=False)["surprise"].apply(_z).reindex(ev.index)