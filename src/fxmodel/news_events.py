"""Мэдээний үеийн шинжилгээний өгөгдөл (M5).

Нэг 5 минутын лаанд гарсан өндөр нөлөөтэй USD/EUR мэдээнүүдийг нэг "үйл явдал" гэж нэгтгэнэ.
Лаа p = мэдээ гарсан минутыг агуулсан 5 минутын лаа (жишээ нь 12:30 мэдээ -> 12:30–12:35 лаа).

Surprise-ийн EUR/USD-д нөлөөлөх чиглэл:
  ForexFactory-ийн "Usual Effect" ('Actual' greater/less than 'Forecast' is good for currency) ->
  sign_good = +1 (их нь сайн) / −1 (бага нь сайн).  EUR мэдээ: +1, USD мэдээ: −1 (EUR/USD-ийн хувьд).
  s = sign_good × валютын тэмдэг × surprise_z (±5-аар хязгаарласан). s > 0 -> EUR/USD өсөх ёстой.

Зорилтууд (пип):
  R60  : лаа p-ээс эхлэх 60 минутын хүрээ (мэдээний хариу үйлдлийн хэмжээ) — мэдээ гарахаас ӨМНӨ таамаглана
  J    : лаа p-ийн хаалт − нээлт (эхний 5 минутын "үсрэлт")
  POST : лаа p+12-ийн хаалт − лаа p-ийн хаалт (дараагийн 60 минут) — эхний 5 минутын ДАРАА таамаглана
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PIP = 1e-4
EFF_RE = r"Usual Effect:\s*'Actual'\s*(greater|less) than 'Forecast' is good for currency"


def usual_effect(raw_csv) -> dict:
    r = pd.read_csv(raw_csv, dtype=str, keep_default_na=False)
    r = r[r["Currency"].isin(["USD", "EUR"])]
    eff = r["Detail"].str.extract(EFF_RE, flags=re.I)[0]
    m = r.assign(eff=eff).dropna(subset=["eff"]).groupby(["Currency", "Event"])["eff"].agg(lambda s: s.mode().iat[0])
    return {k: (1 if v.lower() == "greater" else -1) for k, v in m.items()}


def build_events(m5: pd.DataFrame, ev: pd.DataFrame, effect: dict, z_clip: float = 5.0) -> pd.DataFrame:
    ev = ev[(ev["impact"] == "high") & ev["currency"].isin(["USD", "EUR"])].copy()
    ev["bar"] = ev["ts_utc"].dt.floor("5min")
    ev["good"] = [effect.get((c, t), 0) for c, t in zip(ev["currency"], ev["title"])]
    ev["s"] = ev["good"] * np.where(ev["currency"] == "EUR", 1, -1) * ev["surprise_z"].clip(-z_clip, z_clip)
    idx = m5.index.as_unit("ns")
    o, h, l, c = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
    groups = list(ev.groupby("bar"))
    # Индекс хайлтыг нэг удаа, вектороор (цагийн нарийвчлал ижил байх ёстой — эс бөгөөс удаан замаар явна)
    pos = idx.get_indexer(pd.DatetimeIndex([b for b, _ in groups]).as_unit("ns"))
    rows = []
    for (bar, g), p in zip(groups, pos):
        if p < 300 or p + 13 >= len(idx):
            continue
        # Мэдээний өмнөх 1ц ба дараах 1ц тасралтгүй 5 минутын лаатай байх
        if (idx[p + 12] - idx[p]) != pd.Timedelta(minutes=60) or (idx[p] - idx[p - 12]) != pd.Timedelta(minutes=60):
            continue
        s_vals = g["s"].dropna()
        key = g.loc[g["s"].abs().fillna(-1).idxmax()] if len(s_vals) else g.iloc[0]
        rows.append(dict(
            bar=bar, p=p, titles="|".join(sorted(g["title"])), n_events=len(g),
            n_usd=int((g["currency"] == "USD").sum()), n_eur=int((g["currency"] == "EUR").sum()),
            S=float(s_vals.sum()) if len(s_vals) else np.nan, S_max=float(key["s"]) if len(s_vals) else np.nan,
            n_with_surprise=int(len(s_vals)),
            n_pos=int((s_vals >= 0.5).sum()), n_neg=int((s_vals <= -0.5).sum()),   # зөрчилтэй surprise-ийг илрүүлэх
            key_title=str(key["title"]), key_currency=str(key["currency"]),
            pre_ret30=(c[p - 1] - c[p - 7]) / PIP,                                   # мэдээний өмнөх 30 минут
            R60=(h[p:p + 12].max() - l[p:p + 12].min()) / PIP,
            J=(c[p] - o[p]) / PIP,
            POST=(c[p + 12] - c[p]) / PIP,
            pre_r60=(h[p - 12:p].max() - l[p - 12:p].min()) / PIP,
            pre_r24h=(h[max(0, p - 288):p].max() - l[max(0, p - 288):p].min()) / PIP,
            hour=bar.hour + bar.minute / 60, dow=bar.dayofweek,
        ))
    return pd.DataFrame(rows).set_index("bar")


def type_effect(train: pd.DataFrame, col: str, prior_weight: float = 5.0) -> tuple[dict, float]:
    """Мэдээний төрөл бүрийн түүхэн дундаж (train-ээс), цөөн тохиолдолтойг ерөнхий дундаж руу татна."""
    rows = []
    for t, v in zip(train["titles"], train[col]):
        for k in t.split("|"):
            rows.append((k, v))
    d = pd.DataFrame(rows, columns=["t", "v"])
    glob = float(d["v"].mean())
    agg = d.groupby("t")["v"].agg(["sum", "count"])
    return ((agg["sum"] + prior_weight * glob) / (agg["count"] + prior_weight)).to_dict(), glob


def apply_type_effect(df: pd.DataFrame, table: dict, glob: float) -> pd.Series:
    """Нэг лаанд хэд хэдэн мэдээ байвал хамгийн их нөлөөтэйг нь авна."""
    return df["titles"].map(lambda t: max(table.get(k, glob) for k in t.split("|")))


# ---------------- Бодит цагийн систем: мэдээний профайл ба post-mortem ----------------
def news_profile(E: pd.DataFrame, D: pd.DataFrame, effect: dict, prior_n: float = 10.0) -> dict:
    """Мэдээний төрөл бүрийн түүхэн профайл.

    E: build_events + seasonal (лог хэмжээ) бүхий бүх үйл явдал (reports/news_events.csv)
    D: surprise-тэй үйл явдал, hit баганатай (reports/news_errors_events.csv)
    Буцаана: {title: {n, r60_med, r60_q10, r60_q90, mult, hit5, n_hit, good}} + "_global".
    mult = тухайн цагийн ердийн 60 минутын хүрээнээс хэд дахин том (цөөн тохиолдолтойг ерөнхий рүү татсан).
    hit5 = surprise-ийн чиглэлтэй эхний 5 минутын хөдөлгөөн таарах хувь (prior_n-ээр ерөнхий рүү татсан).
    """
    rows = [(t, r60, r60 / np.exp(s)) for ts, r60, s in zip(E["titles"], E["R60"], E["seasonal"]) for t in ts.split("|")]
    A = pd.DataFrame(rows, columns=["t", "r60", "ratio"])
    g_ratio = float(np.exp(np.log(A["ratio"]).mean()))
    g_hit = float(D["hit"].mean())
    hit = D.groupby("key_title")["hit"].agg(["sum", "count"])
    prof = {}
    for t, a in A.groupby("t"):
        lr = np.log(a["ratio"])
        mult = float(np.exp((lr.sum() + 5 * np.log(g_ratio)) / (len(lr) + 5)))
        h = hit.loc[t] if t in hit.index else None
        prof[t] = dict(n=int(len(a)), r60_med=round(float(a["r60"].median()), 1),
                       r60_q10=round(float(a["r60"].quantile(0.1)), 1), r60_q90=round(float(a["r60"].quantile(0.9)), 1),
                       mult=round(mult, 2),
                       hit5=round(float((h["sum"] + prior_n * g_hit) / (h["count"] + prior_n)) if h is not None else g_hit, 3),
                       n_hit=int(h["count"]) if h is not None else 0)
    for (cur, t), v in effect.items():
        if t in prof:
            prof[t]["good"] = v
            prof[t].setdefault("currency", cur)
    prof["_global"] = dict(n=int(len(E)), r60_med=round(float(E["R60"].median()), 1),
                           r60_q10=round(float(E["R60"].quantile(0.1)), 1), r60_q90=round(float(E["R60"].quantile(0.9)), 1),
                           mult=round(g_ratio, 2), hit5=round(g_hit, 3))
    return prof


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")
_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def parse_value(s) -> float:
    """ForexFactory-ийн утга ('0.3%', '-12.5K', '<0.10%', '1.2B') -> тоо. Уншигдахгүй бол NaN."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return np.nan
    v = float(m.group())
    tail = s[m.end():m.end() + 1].upper()
    return v * _SCALE.get(tail, 1.0)


def surprise_sign(currency: str, title: str, actual, forecast, effect: dict) -> float:
    """EUR/USD-д нөлөөлөх surprise-ийн тэмдэг: +1 (EUR/USD өсөх ёстой), −1, 0 (таарсан), NaN (мэдэгдэхгүй)."""
    a, f = parse_value(actual), parse_value(forecast)
    good = effect.get((currency, title))
    if np.isnan(a) or np.isnan(f) or not good:
        return np.nan
    return float(np.sign(a - f) * good * (1 if currency == "EUR" else -1))


def postmortem(m1: pd.DataFrame, events: pd.DataFrame, profile: dict, effect: dict, now: pd.Timestamp,
               lookback_h: int = 72) -> list[dict]:
    """Өнгөрсөн өндөр нөлөөтэй мэдээ бүрийн дараах автомат шинжилгээ (урьдчилан тогтоосон ангиллаар).

    Хэмжээ: бодит 60 минутын хүрээг төрлийн түүхэн 10–90%-тай харьцуулна.
    Чиглэл (actual, forecast байгаа үед): эхний 5 минутын хөдөлгөөн surprise-ийн чиглэлтэй таарсан эсэх.
    Таараагүй бол шалтгааныг алдааны шинжилгээний (MODEL.md 2а) дарааллаар нэрлэнэ.
    """
    ev = events[(events["impact"] == "high") & events["currency"].isin(["USD", "EUR"])].copy()
    ev = ev[(ev["ts_utc"] <= now - pd.Timedelta(minutes=5)) & (ev["ts_utc"] >= now - pd.Timedelta(hours=lookback_h))]
    if ev.empty or m1 is None or m1.empty:
        return []
    for c in ("actual", "forecast"):
        if c not in ev:
            ev[c] = None
    ev["bar"] = ev["ts_utc"].dt.floor("5min")
    idx = m1.index
    glob = profile.get("_global", {})
    out = []
    for bar, g in ev.groupby("bar"):
        w = m1[(idx >= bar) & (idx < bar + pd.Timedelta(minutes=60))]
        if len(w) < 5:
            continue
        J = (w["close"].iloc[4] - w["open"].iloc[0]) / PIP
        R60 = (w["high"].max() - w["low"].min()) / PIP
        complete = (w.index[-1] - bar) >= pd.Timedelta(minutes=55)
        profs = [profile.get(t, glob) for t in g["title"]]
        key = max(profs, key=lambda p: p.get("r60_med", 0))
        signs = [surprise_sign(c, t, a, f, effect) for c, t, a, f in zip(g["currency"], g["title"], g["actual"], g["forecast"])]
        known = [s for s in signs if not np.isnan(s) and s != 0]
        conflict = len(known) > 1 and len(set(known)) > 1
        if not complete:
            size = "хэмжиж байна (60 минут болоогүй)"
        elif R60 > key["r60_q90"]:
            size = "хүлээгдсэнээс том"
        elif R60 < key["r60_q10"]:
            size = "хүлээгдсэнээс бага"
        else:
            size = "хүлээгдсэн хүрээнд"
        hit, reason = None, None
        if not known:
            reason = "surprise тодорхойгүй (actual/forecast алга эсвэл таарсан)"
        elif conflict:
            reason = "зөрчилтэй surprise (эсрэг чиглэлтэй мэдээнүүд)"
        elif abs(J) < 2:
            reason = "зах зээл хариу үзүүлээгүй (|J| < 2 пип)"
        else:
            hit = bool(np.sign(J) == known[0])
            if not hit:
                h5 = min(p.get("hit5", 1) for p in profs)
                reason = ("EUR/USD-д сул хамааралтай мэдээ" if h5 < 0.62 else
                          "тайлбарлагдаагүй (зах зээлийн хүлээлт, засвар эсвэл горим)")
        out.append(dict(time=int(bar.timestamp()), titles=[f"{c} · {t}" for c, t in zip(g["currency"], g["title"])],
                        surprise_dir=None if not known else (int(known[0]) if not conflict else 0),
                        J=round(float(J), 1), R60=round(float(R60), 1), expected=[key["r60_q10"], key["r60_med"], key["r60_q90"]],
                        size=size, hit=hit, reason=reason, complete=bool(complete)))
    return out[::-1]
