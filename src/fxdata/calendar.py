"""Эдийн засгийн мэдээний календарь унших, UTC болгох, утгыг тоо болгох, surprise тооцох.

Гол эрсдэл: ЦАГ. HuggingFace дээрх ForexFactory 2007–2025 датасетэд 3 асуудал илэрсэн
(шалгалт: scripts/validate_news_timing.py, тайлан: pipeline_report.json -> calendar):

1. Файлд бичсэн UTC offset 2023 оноос хойш зуны улиралд БУРУУ (+04:30). Иран 2023 оноос
   зуны цаг хэрэглэхээ больсон тул бодит offset +03:30 байсан. Иймээс файлын offset-ийг
   үл тоомсорлож, хананы цагийг (wall time) tzdata-аар Asia/Tehran бүсээр дахин хөрвүүлнэ.
   Үр дүнд нь 2023–2025 оны зуны мэдээ 1 цагаар эрт буухаа больсон.
2. USD/EUR өндөр нөлөөтэй мэдээний ~40–50% нь цаггүй (00:00:00 эсвэл 23:59:59 гэж
   бичигдсэн). Үүнд NFP, CPI, Retail Sales, FOMC зэрэг хамгийн чухал мэдээнүүд орно. Ийм мөрийг
   хуучин код өмнөх өдрийн 19:30/20:30 UTC-д оноож байсан тул мэдээний actual-ийг
   ~16 цагийн өмнө "мэдэж" байгаа мэт look-ahead алдаа үүсгэж байв.
   Шийдэл: мэдээ тогтмол цагт гардаг тул цагийг (а) албан ёсны хуваарь (SCHEDULE),
   (б) тухайн мэдээ ижил онд цагтайгаар гарсан тохиолдлууд, (в) бүх жилийн давамгай цаг гэсэн
   дарааллаар сэргээнэ. Сэргээж чадаагүй мөрийг (илтгэл, уулзалт, сонгууль) цаг тодорхойгүй
   гэж үзээд лаатай холбохгүй. Мөр бүрт time_source баганаар эх сурвалжийг нь тэмдэглэнэ.
3. Календарь 2025-04-07-нд дуусдаг. Үүнээс хойших лаанд мэдээний шинж хүчингүй.
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


# Мэдээ гарах улсын цагийн бүс (хуваарийг орон нутгийн цагаар бичнэ — зуны цагийг tzdata тооцно)
HOME_TZ = {"USD": "America/New_York", "EUR": "Europe/Berlin"}

# Албан ёсны хуваарь: (валют, гарчгийн regex, орон нутгийн цаг, эхлэх огноо, дуусах огноо)
# Эх сурвалж: BLS/Census/BEA/ISM/S&P Global/ECB/Fed-ийн нийтэлсэн хуваарь.
# Хоёрдмол утгатай (илтгэл, уулзалт, сонгууль) мэдээг энд оруулаагүй.
_US_0830 = (r"^(Non-Farm Employment Change|Unemployment Rate|Average Hourly Earnings m/m|Unemployment Claims|"
            r"(Core )?Retail Sales m/m|(Core )?CPI [my]/[my]|(Core )?PPI m/m|Building Permits|Housing Starts|"
            r"Core PCE Price Index m/m|Personal (Spending|Income) m/m|Trade Balance|(Advance|Prelim|Final) GDP q/q|"
            r"(Core )?Durable Goods Orders m/m|Empire State Manufacturing Index|Employment Cost Index q/q|Import Prices m/m)$")
SCHEDULE = [
    ("USD", _US_0830, "08:30", None, None),
    ("USD", r"^Philly Fed Manufacturing Index$", "10:00", None, "2016-01-01"),
    ("USD", r"^Philly Fed Manufacturing Index$", "08:30", "2016-01-01", None),
    ("USD", r"^ADP Non-Farm Employment Change$", "08:15", None, None),
    ("USD", r"^(ISM (Services|Manufacturing) PMI|JOLTS Job Openings|CB Consumer Confidence|New Home Sales|"
            r"Existing Home Sales|(Prelim|Revised) UoM (Consumer Sentiment|Inflation Expectations))$", "10:00", None, None),
    ("USD", r"^Flash (Services|Manufacturing) PMI$", "09:45", None, None),
    ("USD", r"^Crude Oil Inventories$", "10:30", None, None),
    ("USD", r"^(FOMC Statement|Federal Funds Rate|FOMC Economic Projections)$", "14:00", "2013-01-01", None),
    ("USD", r"^FOMC Press Conference$", "14:30", "2013-01-01", None),
    ("USD", r"^FOMC Meeting Minutes$", "14:00", None, None),
    ("EUR", r"^(Main Refinancing Rate|Monetary Policy Statement|Deposit Facility Rate)$", "13:45", None, "2022-07-21"),
    ("EUR", r"^(Main Refinancing Rate|Monetary Policy Statement|Deposit Facility Rate)$", "14:15", "2022-07-21", None),
    ("EUR", r"^ECB Press Conference$", "14:30", None, "2022-07-21"),
    ("EUR", r"^ECB Press Conference$", "14:45", "2022-07-21", None),
    ("EUR", r"^German Flash (Services|Manufacturing) PMI$", "09:30", None, None),
    ("EUR", r"^French Flash (Services|Manufacturing) PMI$", "09:15", None, None),
    ("EUR", r"^Flash (Services|Manufacturing) PMI$", "10:00", None, None),
    ("EUR", r"^(CPI Flash Estimate y/y|Core CPI Flash Estimate y/y)$", "11:00", None, None),
    ("EUR", r"^German ifo Business Climate$", "10:00", None, None),
    ("EUR", r"^(German )?ZEW Economic Sentiment$", "11:00", None, None),
]
PLACEHOLDER_TIMES = ("00:00:00", "23:59:59")


def _scheduled_time(currency: str, title: str, date: pd.Timestamp):
    for cur, pat, hhmm, since, until in SCHEDULE:
        if cur != currency or not re.match(pat, title):
            continue
        if since and date < pd.Timestamp(since):
            continue
        if until and date >= pd.Timestamp(until):
            continue
        return hhmm
    return None


def _learned_schedule(known: pd.DataFrame) -> tuple[dict, dict]:
    """Цагтай мөрүүдээс (валют, гарчиг[, он]) бүрийн давамгай орон нутгийн цагийг сурна."""
    by_year, overall = {}, {}
    for (cur, title, yr), g in known.groupby(["currency", "title", "year"]):
        vc = g["local_hhmm"].value_counts()
        if vc.iloc[0] >= 2 and vc.iloc[0] / len(g) >= 0.6:
            by_year[(cur, title, yr)] = vc.index[0]
    for (cur, title), g in known.groupby(["currency", "title"]):
        vc = g["local_hhmm"].value_counts()
        if vc.iloc[0] >= 5 and vc.iloc[0] / len(g) >= 0.7:
            overall[(cur, title)] = vc.index[0]
    return by_year, overall


def _localize_wall(wall: pd.Series, tz: str, fallback_offset: pd.Series | None = None) -> pd.Series:
    """Хананы цагийг tz бүсээр UTC болгоно. Зуны цагийн шилжилтийн цоорхой/давхцал дээр
    файл дахь offset-оор нөхнө."""
    loc = wall.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    if fallback_offset is not None:
        miss = loc.isna() & wall.notna() & fallback_offset.notna()
        if miss.any():
            txt = wall[miss].dt.strftime("%Y-%m-%dT%H:%M:%S") + fallback_offset[miss]
            loc[miss] = pd.to_datetime(txt, utc=True, errors="coerce")
    return loc


def load_calendar(path, source_tz: str, currencies=("USD", "EUR")) -> tuple[pd.DataFrame, dict]:
    """CSV календарийг уншиж UTC болгоно. (events, log) буцаана.

    source_tz: CSV-ийн ХАНАНЫ цаг ямар бүсээр бичигдсэн ('Asia/Tehran', 'America/New_York', 'UTC' ...).
    Файлд UTC offset бичигдсэн байсан ч түүнд итгэхгүй — wall time-ийг source_tz-ээр дахин хөрвүүлнэ.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    cols = {c.strip().lower(): c for c in raw.columns}
    m = {k: _find(cols, k) for k in COLUMN_CANDIDATES}
    for need in ("datetime", "currency", "impact", "title"):
        if m[need] is None:
            raise ValueError(f"Календарьт '{need}' багана олдсонгүй. Байгаа баганууд: {list(raw.columns)}")

    log = {"rows_raw": len(raw)}
    dt_text = raw[m["datetime"]].str.strip()
    if m["time"] is not None and m["time"] != m["datetime"] and dt_text.str.contains(":").mean() < 0.5:
        dt_text = dt_text + " " + raw[m["time"]].str.strip()

    ev = pd.DataFrame({
        "currency": raw[m["currency"]].str.strip().str.upper(),
        "impact": raw[m["impact"]].map(normalize_impact),
        "title": raw[m["title"]].str.strip(),
        "actual_raw": raw[m["actual"]] if m["actual"] else "",
        "forecast_raw": raw[m["forecast"]] if m["forecast"] else "",
        "previous_raw": raw[m["previous"]] if m["previous"] else "",
        "raw_time_text": dt_text,
    })
    ev = ev[ev["currency"].isin(currencies)].copy()
    txt = ev["raw_time_text"]

    # Текстэн тэмдэглэгээ ("All Day", "Tentative", "Day 1") -> цаг тодорхойгүй
    textual = txt.str.contains(r"all day|tentative|day \d", case=False, regex=True)
    off_re = r"([+-]\d{2}:?\d{2}|Z)$"
    offset = txt.str.extract(off_re, expand=False)
    wall_txt = txt.str.replace(off_re, "", regex=True).str.replace("T", " ", regex=False).str.strip()
    wall = pd.to_datetime(wall_txt.where(~textual), errors="coerce", format="mixed")
    has_clock = wall_txt.str.contains(r"\d{1,2}:\d{2}", regex=True)
    placeholder = wall.dt.strftime("%H:%M:%S").isin(PLACEHOLDER_TIMES) | ~has_clock
    log["time_textual"] = int(textual.sum())
    log["time_placeholder"] = int((placeholder & wall.notna()).sum())

    # 1) Жинхэнэ цагтай мөр: wall time -> source_tz (tzdata) -> UTC
    real = wall.notna() & ~placeholder
    ts = pd.Series(pd.NaT, index=ev.index, dtype="datetime64[ns, UTC]")
    ts[real] = _localize_wall(wall[real], source_tz, offset[real])
    if offset.notna().any():
        file_utc = pd.to_datetime(txt[real & offset.notna()], utc=True, errors="coerce", format="mixed")
        log["file_offset_disagrees_with_tzdata"] = int((file_utc != ts[file_utc.index]).sum())
    src = pd.Series("unknown", index=ev.index)
    src[real & ts.notna()] = "original"

    # 2) Цаггүй мөр: огноо + хуваарь (мэдээ гарах улсын орон нутгийн цагаар)
    local_hhmm = pd.Series(index=ev.index, dtype=object)
    for cur, tz in HOME_TZ.items():
        sel = real & ts.notna() & (ev["currency"] == cur)
        local_hhmm[sel] = ts[sel].dt.tz_convert(tz).dt.strftime("%H:%M")
    known = ev.loc[local_hhmm.notna(), ["currency", "title"]].assign(
        year=ts[local_hhmm.notna()].dt.year, local_hhmm=local_hhmm[local_hhmm.notna()])
    by_year, overall = _learned_schedule(known)

    need = placeholder & wall.notna() & ev["currency"].isin(list(HOME_TZ))
    imputed = {}
    for i in ev.index[need]:
        cur, title, day = ev.at[i, "currency"], ev.at[i, "title"], wall.at[i].normalize()
        hhmm, how = _scheduled_time(cur, title, day), "schedule"
        if hhmm is None:
            hhmm, how = by_year.get((cur, title, day.year)), "learned_year"
        if hhmm is None:
            hhmm, how = overall.get((cur, title)), "learned_all"
        if hhmm is not None:
            imputed[i] = (day + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[3:])), how)
    for cur, tz in HOME_TZ.items():
        idx = [i for i in imputed if ev.at[i, "currency"] == cur]
        if idx:
            w = pd.Series([imputed[i][0] for i in idx], index=idx)
            ts[idx] = _localize_wall(w, tz)
            src[idx] = [imputed[i][1] for i in idx]
    ev["ts_utc"] = ts
    ev["time_source"] = src
    log["time_imputed"] = {k: int(v) for k, v in src[src.isin(["schedule", "learned_year", "learned_all"])].value_counts().items()}
    log["time_unknown_dropped"] = int(ev["ts_utc"].isna().sum())
    log["time_unknown_high_impact"] = int((ev["ts_utc"].isna() & (ev["impact"] == "high")).sum())
    log["calendar_end_utc"] = str(ev["ts_utc"].max())

    ev = ev[ev["ts_utc"].notna()].copy()
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