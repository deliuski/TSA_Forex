"""2025-04-07-ноос хойших ForexFactory календарийг (ehsanrs2/forexfactory-scraper-ийн гаралт) хуучин
HuggingFace файлтай нэгтгэх.

Гол эрсдэл нь дахиад ЦАГ. Скрейпер нь ForexFactory хуудсан дээр ХАРАГДСАН цагийг `--tz`-ээр бичсэн гэж үздэг.
Гэвч хуудас ямар бүсээр цаг харуулах нь сайтын тохиргоо, байршлаас хамаарна. Тиймээс файлын offset-д итгэхгүй.
Тогтмол хуваарьтай мэдээнүүдийн (NFP 08:30 New York, ECB 14:15 Frankfurt гэх мэт) бодит UTC цагийг хуудасны
хананы цагтай харьцуулж, хуудасны цагийн бүсийг таана (`infer_page_tz`).
Дараа нь мөр бүрийг UTC болгоод хуучин файлын хэлбэрээр (Asia/Tehran хананы цаг) бичнэ.
Ингэснээр `calendar.load_calendar(..., "Asia/Tehran")` хоёр хэсгийг ижил аргаар уншина.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .calendar import HOME_TZ, _scheduled_time

CANDIDATE_TZ = ["UTC", "America/New_York", "America/Chicago", "America/Los_Angeles", "Europe/London",
                "Europe/Berlin", "Asia/Tehran", "Asia/Ulaanbaatar", "Asia/Shanghai", "Asia/Tokyo", "Australia/Sydney"]
OLD_TZ = "Asia/Tehran"
IMPACT_TXT = {"high": "High Impact Expected", "medium": "Medium Impact Expected", "low": "Low Impact Expected"}


def _wall(series: pd.Series) -> pd.Series:
    """'2025-05-02T08:30:00-04:00' -> хананы цаг (offset-гүй). Цаггүй/текст -> NaT."""
    s = series.fillna("").astype(str).str.strip()
    s = s.str.replace(r"([+-]\d{2}:?\d{2}|Z)$", "", regex=True).str.replace("T", " ", regex=False)
    return pd.to_datetime(s.where(s.str.contains(r"\d{1,2}:\d{2}")), errors="coerce", format="mixed")


def true_scheduled_utc(currency: str, title: str, date: pd.Timestamp) -> pd.Timestamp | None:
    """Албан ёсны хуваарьт мэдээний бодит UTC цаг (тухайн өдөр нь орон нутгийн огноо)."""
    hhmm = _scheduled_time(currency, title, date)
    if hhmm is None or currency not in HOME_TZ:
        return None
    local = pd.Timestamp(f"{date:%Y-%m-%d} {hhmm}").tz_localize(HOME_TZ[currency], ambiguous=False, nonexistent="shift_forward")
    return local.tz_convert("UTC")


def infer_page_tz(df: pd.DataFrame) -> dict:
    """Хуудасны хананы цаг аль бүсийнх вэ? Хуваарьт мэдээний бодит UTC-тэй хамгийн олон таарах бүсийг сонгоно.

    df: currency, title, wall (хуудасны хананы цаг).
    Буцаана: {tz, matches, n, share, ranking}
    """
    rows = []
    for cur, title, w in zip(df["currency"], df["title"], df["wall"]):
        if pd.isna(w):
            continue
        # Орон нутгийн огноо хуудасны огнооноос ±1 өдөр зөрж болно — гурвууланг нь шалгана
        for dd in (0, -1, 1):
            u = true_scheduled_utc(cur, title, (w + pd.Timedelta(days=dd)).normalize())
            if u is not None:
                rows.append((w, u))
                break
    if not rows:
        return dict(tz=None, matches=0, n=0, share=0.0, ranking={})
    W = pd.Series([r[0] for r in rows])
    U = pd.Series([r[1] for r in rows])
    score = {}
    for tz in CANDIDATE_TZ:
        loc = W.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
        score[tz] = int((loc == U).sum())
    best = max(score, key=score.get)
    return dict(tz=best, matches=score[best], n=len(rows), share=score[best] / len(rows),
                ranking=dict(sorted(score.items(), key=lambda kv: -kv[1])[:4]))


def to_old_format(new: pd.DataFrame, page_tz: str, detail_map: dict) -> pd.DataFrame:
    """Скрейперийн legacy багануудыг хуучин HuggingFace файлын хэлбэрт (Asia/Tehran хананы цаг) хөрвүүлнэ.
    Цаггүй мөрийг (All Day, Tentative) '00:00:00' буюу хуучин файлын цаггүй тэмдэглэгээгээр бичнэ — loader
    хуваариас сэргээнэ."""
    wall = _wall(new["DateTime"])
    date_only = pd.to_datetime(new["DateTime"].astype(str).str[:10], errors="coerce")
    utc = wall.dt.tz_localize(page_tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    teh = utc.dt.tz_convert(OLD_TZ)
    off = teh.dt.strftime("%z").str.replace(r"(\d{2})(\d{2})$", r"\1:\2", regex=True)
    txt = teh.dt.strftime("%Y-%m-%dT%H:%M:%S") + off
    placeholder = date_only.dt.strftime("%Y-%m-%dT00:00:00") + "+03:30"
    out = pd.DataFrame({
        "DateTime": txt.where(utc.notna(), placeholder),
        "Currency": new["Currency"].astype(str).str.strip().str.upper(),
        "Impact": new["Impact"].astype(str).map(lambda x: next((v for k, v in IMPACT_TXT.items() if k in x.lower()), x)),
        "Event": new["Event"].astype(str).str.strip(),
        "Actual": new["Actual"].fillna(""), "Forecast": new["Forecast"].fillna(""), "Previous": new["Previous"].fillna(""),
    })
    out["Detail"] = [detail_map.get((c, e), "") for c, e in zip(out["Currency"], out["Event"])]
    out["_utc"] = utc
    out["_date"] = date_only
    return out[date_only.notna()]


def detail_map_from_old(old: pd.DataFrame) -> dict:
    """Хуучин файлаас (валют, мэдээ) бүрийн Detail ('Usual Effect' агуулсан) — шинэ мөрөнд хуулна."""
    d = old[old["Detail"].str.contains("Usual Effect", na=False)]
    return d.groupby(["Currency", "Event"])["Detail"].last().to_dict()


def merge(old: pd.DataFrame, new_conv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Хуучин файлын сүүлийн огнооноос хойших шинэ мөрүүдийг нэмнэ (давхардсан огноо+валют+мэдээг хасна)."""
    old_date = pd.to_datetime(old["DateTime"].str[:10], errors="coerce")
    last = old_date.max()
    add = new_conv[new_conv["_date"] > last]
    key_old = set(zip(old_date.dt.strftime("%Y-%m-%d"), old["Currency"], old["Event"]))
    add = add[[(d, c, e) not in key_old for d, c, e in zip(add["_date"].dt.strftime("%Y-%m-%d"), add["Currency"], add["Event"])]]
    add = add.drop_duplicates(["DateTime", "Currency", "Event"])
    cols = ["DateTime", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous", "Detail"]
    merged = pd.concat([old[cols], add[cols]], ignore_index=True)
    info = dict(old_rows=len(old), old_last=str(last.date()), added=len(add),
                added_range=[str(add["_date"].min().date()), str(add["_date"].max().date())] if len(add) else None,
                added_usd_eur_high=int(((add["Currency"].isin(["USD", "EUR"])) & add["Impact"].str.startswith("High")).sum()),
                added_without_time=int(add["_utc"].isna().sum()))
    return merged, info
