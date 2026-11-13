"""EUR/USD-ийн дараагийн 24 цагийн хувилбарууд ба магадлалыг бодит цагт гаргах.

Алхам:
  1. Dukascopy-оос сүүлийн ~90 хоногийн 1 цагийн лаа татна (dukascopy-node, Node.js 20+).
  2. ForexFactory-ийн энэ долоо хоногийн календарийг татна (мэдээний ХУВААРЬ).
     Татагдахгүй бол мэдээгүй загвар (live_nonews) руу автоматаар шилжинэ.
  3. Шинж тооцоод хамгийн сүүлийн ХААГДСАН лаан дээр хувилбар бүрийн магадлалыг гаргана.
  4. Үр дүн: консолд хүснэгт, data/live/latest.png график, data/live/predictions.csv бүртгэл.
     24 цаг өнгөрсөн таамаглалын бодит хувилбарыг бүртгэлд нөхөж бичнэ (бодит цагийн үнэлгээ).

Ажиллуулах:
  python scripts/predict_live.py            # нэг удаа
  python scripts/predict_live.py --loop     # цаг бүрийн 2 дахь минутад давтан ажиллана
  python scripts/predict_live.py --candles-csv X.csv --calendar-json Y.json --now "2025-04-04 15:00"  # офлайн тест

Анхаар: энэ бол судалгааны прототип. Санхүүгийн зөвлөгөө биш. Test үеийн чиглэлийн AUC ≈ 0.50 тул
"өсөх/унах" тал дээр бараг мэдээлэлгүй. Хөдөлгөөний хэмжээ (range vs том хөдөлгөөн) дээр л ажиллана.
"""
import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxdata import merge, prices  # noqa: E402
from fxmodel.features import atr, build_features  # noqa: E402
from fxmodel.scenarios import future_paths  # noqa: E402
from fxmodel.volatility import har_features  # noqa: E402
from fxmodel.paths import PathLibrary  # noqa: E402
from fxmodel.news_events import postmortem  # noqa: E402

LIVE = ROOT / "data" / "live"
MOD = ROOT / "models"
FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
DUKASCOPY_NODE = "dukascopy-node@1.51.1"
IMPACT = {"High": "high", "Medium": "medium", "Low": "low", "Holiday": "holiday"}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ---------------- Өгөгдөл татах ----------------
def fetch_candles(days: int = 90) -> pd.DataFrame:
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx олдсонгүй. Node.js 20+ суулгана уу: https://nodejs.org")
    LIVE.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    start = (now - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    name = "eurusd_h1_live"
    cmd = [npx, "-y", DUKASCOPY_NODE, "-i", "eurusd", "-from", start, "-to", end,
           "-t", "h1", "-p", "bid", "-v", "-f", "csv", "-dir", str(LIVE), "-fn", name, "--retries", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Dukascopy-оос татаж чадсангүй:\n{r.stderr[-1500:]}")
    return prices.load_dukascopy_csv(LIVE / f"{name}.csv")


def parse_ff_json(items: list) -> pd.DataFrame:
    rows = []
    for it in items:
        cur = str(it.get("country", "")).upper()
        if cur not in ("USD", "EUR"):
            continue
        ts = pd.to_datetime(it.get("date"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        rows.append(dict(ts_utc=ts, currency=cur, impact=IMPACT.get(it.get("impact"), "unknown"),
                         title=str(it.get("title", "")).strip(), surprise_z=np.nan,
                         actual=it.get("actual") or None, forecast=it.get("forecast") or None))
    return pd.DataFrame(rows, columns=["ts_utc", "currency", "impact", "title", "surprise_z", "actual", "forecast"])


def fetch_m1(days: int = 3) -> pd.DataFrame | None:
    """Сүүлийн хэдэн өдрийн M1 (мэдээний post-mortem-д). Татагдахгүй бол None."""
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        return None
    now = pd.Timestamp.now(tz="UTC")
    name = "eurusd_m1_live"
    cmd = [npx, "-y", DUKASCOPY_NODE, "-i", "eurusd", "-from", (now - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
           "-to", (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), "-t", "m1", "-p", "bid", "-v", "-f", "csv",
           "-dir", str(LIVE), "-fn", name, "--retries", "3"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[анхаар] M1 татагдсангүй, post-mortem алгасна.")
        return None
    return prices.load_dukascopy_csv(LIVE / f"{name}.csv")


_PROFILE = None


def news_profile():
    """models/news_profile.json (scripts/build_news_profile.py). Байхгүй бол ({}, {})."""
    global _PROFILE
    if _PROFILE is None:
        f = MOD / "news_profile.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            _PROFILE = (d["profile"], {(c, t): v for c, t, v in d["effect"]})
        else:
            _PROFILE = ({}, {})
    return _PROFILE


def news_postmortem(m1: pd.DataFrame | None, events: pd.DataFrame | None, now: pd.Timestamp) -> dict:
    """Сүүлийн 72 цагийн мэдээний post-mortem + түүхийг data/live/postmortem.csv-д хуримтлуулна."""
    prof, eff = news_profile()
    if events is None or m1 is None or not prof:
        return {"items": [], "summary": {}}
    # ForexFactory-ийн үнэгүй JSON-д "actual" байдаггүй. data/live/actuals.csv (title,date,actual) бөглөвөл
    # чиглэлийн шинжилгээ хийгдэнэ; бөглөөгүй бол зөвхөн хэмжээний шинжилгээ.
    ev = events.copy()
    fa = LIVE / "actuals.csv"
    if fa.exists():
        a = pd.read_csv(fa, dtype=str)
        a["day"] = pd.to_datetime(a["date"]).dt.strftime("%Y-%m-%d")
        ev["day"] = ev["ts_utc"].dt.strftime("%Y-%m-%d")
        ev = ev.merge(a[["title", "day", "actual"]].rename(columns={"actual": "actual_m"}), on=["title", "day"], how="left")
        ev["actual"] = ev["actual"].where(ev["actual"].notna(), ev["actual_m"])
    items = postmortem(m1[m1.index < now], ev, prof, eff, now)
    path = LIVE / "postmortem.csv"
    done = pd.DataFrame([dict(i, titles=" | ".join(i["titles"]), expected=json.dumps(i["expected"]))
                         for i in items if i["complete"]])
    if path.exists():
        done = pd.concat([pd.read_csv(path), done]).drop_duplicates("time", keep="last") if len(done) else pd.read_csv(path)
    if len(done):
        done.sort_values("time").to_csv(path, index=False)
    summ = {}
    done = done[done["time"] < int(now.timestamp())] if len(done) else done
    if len(done):
        h = done["hit"].dropna()
        h = h.map(lambda v: v if isinstance(v, bool) else str(v) == "True")
        summ = dict(n=int(len(done)), n_dir=int(len(h)), hit=float(h.mean()) if len(h) else None,
                    bigger=int((done["size"] == "хүлээгдсэнээс том").sum()),
                    inside=int((done["size"] == "хүлээгдсэн хүрээнд").sum()),
                    smaller=int((done["size"] == "хүлээгдсэнээс бага").sum()))
    return {"items": items, "summary": summ}


def fetch_calendar() -> pd.DataFrame | None:
    """Энэ долоо хоногийн календарь. Өмнөх долоо хоногуудыг кэшэд хуримтлуулна
    (hours_since_event-д хэрэгтэй)."""
    cache = LIVE / "calendar_cache.csv"
    try:
        req = urllib.request.Request(FF_URL, headers={"User-Agent": "fx-thesis-research/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            new = parse_ff_json(json.load(resp))
    except Exception as e:  # noqa: BLE001
        print(f"[анхаар] Календарь татагдсангүй ({e}). Мэдээгүй загвар ашиглана.")
        return None
    if cache.exists():
        old = pd.read_csv(cache, parse_dates=["ts_utc"])
        old["ts_utc"] = pd.to_datetime(old["ts_utc"], utc=True)
        new = pd.concat([old, new]).drop_duplicates(["ts_utc", "currency", "title"], keep="last")
    new = new.sort_values("ts_utc")
    new.to_csv(cache, index=False)
    return new


# ---------------- Боломжит замууд ----------------
_LIB = None


def path_library():
    """Түүхэн замын сан (data/processed/eurusd_h1_with_news.parquet-аас нэг удаа үүсгэнэ)."""
    global _LIB
    if _LIB is None:
        f = ROOT / "data" / "processed" / "eurusd_h1_with_news.parquet"
        if not f.exists():
            return None
        hist = pd.read_parquet(f)
        _LIB = PathLibrary(24).build(hist, hist["hours_to_next_event"])
    return _LIB


def possible_paths(candles: pd.DataFrame, htn: float, range_q50: float | None, n: int = 200) -> dict | None:
    lib = path_library()
    if lib is None:
        return None
    a = float(atr(candles).iloc[-1])
    logvr = float(np.log(atr(candles).iloc[-1] / atr(candles, 100).iloc[-1]))
    t = candles.index[-1]
    sel = lib.select((t + pd.Timedelta(hours=1)).hour, logvr, htn, n)
    s = lib.scale_factor(sel, a / 1e-4, range_q50)
    c0 = float(candles["close"].iloc[-1])
    conv = lambda m: np.round(c0 + m[sel] * a * s, 5).tolist()  # noqa: E731
    return dict(times=[int((t + pd.Timedelta(hours=k)).timestamp()) for k in range(0, lib.h + 1)],
                close=conv(lib.close), high=conv(lib.high), low=conv(lib.low), n=len(sel), scale=s,
                analog_dates=[str(d)[:10] for d in lib.index[sel[:5]]])


# ---------------- Таамаглал ----------------
def load_models():
    meta = json.loads((MOD / "meta.json").read_text(encoding="utf-8"))
    with open(MOD / "scenario_model.pkl", "rb") as fh:
        sm = pickle.load(fh)
    boosters = {k: lgb.Booster(model_file=str(MOD / f"{k}.txt")) for k in meta["models"]}
    return meta, sm, boosters


def predict(candles: pd.DataFrame, events: pd.DataFrame | None, now: pd.Timestamp):
    meta, sm, boosters = load_models()
    candles = candles[candles.index + pd.Timedelta(hours=1) <= now]   # зөвхөн хаагдсан лаа
    candles = candles[~candles.index.duplicated()].sort_index()
    if len(candles) < 400:
        sys.exit(f"Лаа хэт цөөн ({len(candles)}). Дор хаяж 400 хэрэгтэй.")
    key = "live_news" if events is not None and len(events) else "live_nonews"
    if key == "live_news":
        df = candles.join(merge.news_features(candles, events))
        # Календарийн хамрах хүрээнээс хойш "дараагийн мэдээ" тодорхойгүй
        df.loc[df.index + pd.Timedelta(hours=1) > events["ts_utc"].max(), "hours_to_next_event"] = np.nan
        feats = build_features(df, groups=("A", "D"))
    else:
        feats = build_features(candles, groups=("A",))
    info = meta["models"][key]
    x = feats[info["columns"]].iloc[[-1]]
    prob = boosters[key].predict(x)[0]
    t = candles.index[-1]
    vol = predict_range(feats.join(har_features(candles)), "vol_news" if key == "live_news" else "vol_nonews")
    htn = float(feats["hours_to_next_event"].iloc[-1]) if "hours_to_next_event" in feats else float("nan")
    paths = possible_paths(candles, htn, vol["q50"] if vol else None)
    return dict(made_at=t + pd.Timedelta(hours=1), candle_ts=t, close=float(candles["close"].iloc[-1]),
                atr=float(atr(candles).iloc[-1]), model=key, prob=prob, sm=sm, meta=meta, candles=candles, vol=vol,
                paths=paths)


def predict_range(feats: pd.DataFrame, key: str) -> dict | None:
    """Ирэх 24 цагийн хүрээ (пип): 10/50/90 хувийн quantile. Загвар байхгүй бол None."""
    f = MOD / "vol_meta.json"
    if not f.exists():
        return None
    vm = json.loads(f.read_text(encoding="utf-8"))["models"][key]
    out = {}
    for a, info in vm.items():
        b = lgb.Booster(model_file=str(MOD / info["file"]))
        out[a] = float(np.exp(b.predict(feats[info["columns"]].iloc[[-1]])[0]))
    typical = float(np.exp(feats["har_m"].iloc[-1]))           # сүүлийн 22 өдрийн дундаж (log) хүрээ
    return dict(q10=out["0.1"], q50=out["0.5"], q90=out["0.9"], typical=typical)


def log_prediction(res: dict):
    path = LIVE / "predictions.csv"
    row = {"made_at_utc": res["made_at"], "close": res["close"], "atr": res["atr"], "model": res["model"]}
    row |= {f"p{i}": float(p) for i, p in enumerate(res["prob"])}
    row["realized"] = np.nan
    df = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df]).drop_duplicates("made_at_utc", keep="first")
    df.to_csv(path, index=False)


def score_past(candles: pd.DataFrame, sm, horizon: int):
    """24 цаг өнгөрсөн таамаглалын бодит хувилбарыг нөхөж, бодит цагийн log-loss тооцно."""
    path = LIVE / "predictions.csv"
    if not path.exists():
        return {"n_scored": 0}
    log = pd.read_csv(path)
    paths = future_paths(candles, horizon)
    lab = pd.Series(np.nan, index=paths.index)
    okp = paths.notna().all(axis=1)
    lab[okp] = sm.assign(paths[okp])
    ts = pd.to_datetime(log["made_at_utc"], utc=True) - pd.Timedelta(hours=1)
    todo = log["realized"].isna()
    log.loc[todo, "realized"] = lab.reindex(ts[todo]).to_numpy()
    log.to_csv(path, index=False)
    done = log.dropna(subset=["realized"])
    summary = {"n_scored": int(len(done))}
    if len(done):
        pcols = [c for c in log.columns if c.startswith("p") and c[1:].isdigit()]
        p = done[pcols].to_numpy()
        y = done["realized"].astype(int).to_numpy()
        ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()
        base = -np.log(sm.base_rates[y]).mean()
        print(f"Бодит цагийн үнэлгээ: {len(done)} таамаглал, log-loss {ll:.3f} (baseline {base:.3f})")
        summary |= {"logloss": float(ll), "baseline_logloss": float(base),
                    "top1_hit": float((p.argmax(1) == y).mean())}
    return summary


SESSIONS = {  # орон нутгийн цагаар; зуны цагийг tzdata тооцно
    "tokyo": ("Asia/Tokyo", 9, 18),
    "london": ("Europe/London", 8, 17),
    "newyork": ("America/New_York", 8, 17),
}


def session_hours(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Цаг бүрт аль сесс нээлттэй байгааг (UTC-ээр) буцаана. Зуны цагийн шилжилтийг харгалзана."""
    hours = pd.date_range(start, end, freq="h", tz="UTC")
    out = []
    for t in hours:
        flags = [k for k, (tz, a, b) in SESSIONS.items() if a <= t.tz_convert(tz).hour < b]
        out.append(dict(time=int(t.timestamp()), s=flags))
    return out


def write_json(res: dict, events: pd.DataFrame | None, score: dict, path: Path, n_candles: int = 300):
    """Чартын хуудсанд (web/index.html) зориулсан гаралт."""
    sm, c, pip = res["sm"], res["candles"].iloc[-n_candles:], res["atr"] / 1e-4
    t0 = res["candle_ts"]
    unix = lambda ts: int(pd.Timestamp(ts).timestamp())  # noqa: E731
    fut = [unix(t0 + pd.Timedelta(hours=k)) for k in range(0, len(sm.centroids[0]) + 1)]

    def line(rel):
        vals = np.r_[0.0, rel] * res["atr"] + res["close"]
        return [{"time": t, "value": round(float(v), 5)} for t, v in zip(fut, vals)]

    order = np.argsort(-res["prob"])
    scen = []
    for rank, i in enumerate(order):
        scen.append(dict(id=int(i), rank=rank, name=sm.names[i], prob=float(res["prob"][i]),
                         base=float(sm.base_rates[i]), end_pips=float(sm.centroids[i][-1] * pip),
                         lo_pips=float(sm.q20[i][-1] * pip), hi_pips=float(sm.q80[i][-1] * pip),
                         path=line(sm.centroids[i]), q20=line(sm.q20[i]), q80=line(sm.q80[i])))
    upcoming, news_zone = [], []
    if events is not None and len(events):
        hi = events[events["impact"] == "high"]
        zone = hi[(hi["ts_utc"] >= c.index[0]) & (hi["ts_utc"] <= res["made_at"] + pd.Timedelta(hours=48))]
        news_zone = sorted({unix(t.floor("h")) for t in zone["ts_utc"]})
        e = events[(events["ts_utc"] >= res["made_at"]) & (events["ts_utc"] <= res["made_at"] + pd.Timedelta(hours=48))
                   & (events["impact"] == "high")]
        prof, _ = news_profile()
        glob = prof.get("_global", {})
        upcoming = []
        for r in e.itertuples():
            p = prof.get(r.title)
            upcoming.append(dict(time=unix(r.ts_utc), currency=r.currency, title=r.title, known=p is not None,
                                 r60_med=(p or glob).get("r60_med"), r60_q90=(p or glob).get("r60_q90"),
                                 mult=(p or glob).get("mult"), hit5=(p or glob).get("hit5"), n=(p or {}).get("n", 0)))
    out = dict(
        made_at=unix(res["made_at"]), made_at_text=f"{res['made_at']:%Y-%m-%d %H:%M} UTC",
        close=res["close"], atr_pips=pip, model=res["model"], horizon=len(sm.centroids[0]),
        candles=[dict(time=unix(t), open=r.open, high=r.high, low=r.low, close=r.close) for t, r in zip(c.index, c.itertuples())],
        scenarios=scen, upcoming_events=upcoming, live_score=score, postmortem=res.get("postmortem"), news_hours=news_zone, range=res.get("vol"), paths=res.get("paths"),
        sessions=session_hours(c.index[0], t0 + pd.Timedelta(hours=len(sm.centroids[0]))),
        disclaimer="Судалгааны прототип. Санхүүгийн зөвлөгөө биш. Чиглэлийн AUC ≈ 0.50.",
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def history_json(n: int = 200) -> list:
    """Таамаглалын бүртгэл (сүүлийн n), бодит үр дүнтэй нь."""
    path = LIVE / "predictions.csv"
    if not path.exists():
        return []
    names = json.loads((MOD / "meta.json").read_text(encoding="utf-8"))["names"]
    log = pd.read_csv(path).tail(n)
    pcols = [c for c in log.columns if c.startswith("p") and c[1:].isdigit()]
    rows = []
    for r in log.itertuples():
        p = np.array([getattr(r, c) for c in pcols])
        top = int(p.argmax())
        real = None if pd.isna(r.realized) else int(r.realized)
        rows.append(dict(made_at=str(r.made_at_utc)[:16], model=r.model, top=names[top], top_prob=float(p[top]),
                         realized=None if real is None else names[real],
                         p_realized=None if real is None else float(p[real]), hit=None if real is None else real == top))
    return rows[::-1]


def plot(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sm, prob, c = res["sm"], res["prob"], res["candles"]
    hist = c["close"].iloc[-72:]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(hist.index, hist.values, color="black", lw=1.2, label="EUR/USD (сүүлийн 72 цаг)")
    fut = [res["candle_ts"] + pd.Timedelta(hours=k + 1) for k in range(len(sm.centroids[0]))]
    colors = ["#888888", "#2e7d32", "#c62828", "#1565c0", "#ef6c00"]
    for i in np.argsort(-prob):
        y = res["close"] + sm.centroids[i] * res["atr"]
        lo = res["close"] + sm.q20[i] * res["atr"]
        hi = res["close"] + sm.q80[i] * res["atr"]
        a = 0.15 + 0.85 * prob[i] / prob.max()
        ax.plot(fut, y, color=colors[i], lw=1 + 4 * prob[i], alpha=a, label=f"{sm.names[i]}: {100 * prob[i]:.0f}%")
        if i == int(np.argmax(prob)):
            ax.fill_between(fut, lo, hi, color=colors[i], alpha=0.12)
    ax.axvline(res["made_at"], color="grey", ls="--", lw=0.8)
    ax.set_title(f"EUR/USD — дараагийн 24 цагийн хувилбарууд ({res['made_at']:%Y-%m-%d %H:%M} UTC, загвар: {res['model']})")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_once(args):
    now = pd.Timestamp(args.now, tz="UTC") if args.now else pd.Timestamp.now(tz="UTC")
    if args.candles_csv:
        candles = prices.load_dukascopy_csv(args.candles_csv)
    else:
        candles = fetch_candles()
    if args.calendar_json:
        events = parse_ff_json(json.loads(Path(args.calendar_json).read_text(encoding="utf-8")))
    elif args.no_news:
        events = None
    else:
        events = fetch_calendar()
    res = predict(candles, events, now)
    LIVE.mkdir(parents=True, exist_ok=True)
    m1 = prices.load_dukascopy_csv(args.m1_csv) if args.m1_csv else (None if args.candles_csv else fetch_m1())
    res["postmortem"] = news_postmortem(m1, events, now)
    for it in res["postmortem"]["items"][:5]:
        dirtxt = {True: "таарсан", False: "таараагүй", None: "—"}[it["hit"]]
        print(f"  [мэдээ] {pd.Timestamp(it['time'], unit='s'):%m-%d %H:%M} {', '.join(it['titles'])[:60]}: "
              f"60м хүрээ {it['R60']:.0f} пип ({it['size']}), эхний 5м {it['J']:+.1f} пип, чиглэл {dirtxt}"
              + (f" — {it['reason']}" if it["reason"] else ""))
    log_prediction(res)
    score = score_past(res["candles"], res["sm"], res["meta"]["horizon"])
    plot(res, LIVE / "latest.png")
    write_json(res, events, score, LIVE / "latest.json")
    print(f"\n{res['made_at']:%Y-%m-%d %H:%M} UTC | хаалт {res['close']:.5f} | ATR {res['atr'] / 1e-4:.1f} пип | загвар {res['model']}")
    if res.get("vol"):
        v = res["vol"]
        print(f"  Ирэх 24 цагийн хүрээ: ~{v['q50']:.0f} пип (80%: {v['q10']:.0f}–{v['q90']:.0f}), "
              f"сүүлийн сарын ердийн {v['typical']:.0f} пип")
    sm, pip = res["sm"], res["atr"] / 1e-4
    print("  (Хувилбарууд ATR-д харьцангуй: ATR өндөр үед 'range' ч гэсэн пипээр том байж болно)")
    print(f"  {'Хувилбар':28s} {'магадлал':>8s}  {'ердийн':>6s}  {'24ц-ийн төгсгөл':>16s}  {'хүрээ (20–80%)':>16s}")
    for i in np.argsort(-res["prob"]):
        end = sm.centroids[i][-1] * pip
        lo, hi = sm.q20[i][-1] * pip, sm.q80[i][-1] * pip
        print(f"  {sm.names[i]:28s} {100 * res['prob'][i]:7.1f}%  {100 * sm.base_rates[i]:5.1f}%  {end:+12.0f} пип  {lo:+6.0f}..{hi:+.0f} пип")
    print(f"График: {LIVE / 'latest.png'}  |  Чарт: python scripts/serve.py -> http://127.0.0.1:8000")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="Цаг бүрийн 2 дахь минутад давтана")
    ap.add_argument("--no-news", action="store_true", help="Календарь татахгүй, мэдээгүй загвар ашиглана")
    ap.add_argument("--candles-csv", help="Офлайн тест: dukascopy CSV")
    ap.add_argument("--calendar-json", help="Офлайн тест: ForexFactory JSON")
    ap.add_argument("--m1-csv", help="Офлайн тест: dukascopy-node M1 CSV (мэдээний post-mortem)")
    ap.add_argument("--now", help="Офлайн тест: 'одоо'-гийн UTC цаг")
    args = ap.parse_args()
    if not args.loop:
        run_once(args)
        return
    while True:
        try:
            run_once(args)
        except SystemExit as e:
            print(e)
        except Exception as e:  # noqa: BLE001
            print(f"[алдаа] {e}")
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("h") + pd.Timedelta(hours=1, minutes=2)
        time.sleep(max(30, (nxt - now).total_seconds()))


if __name__ == "__main__":
    main()
