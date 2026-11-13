"""Жинхэнэ ирээдүйн шалгалт (forward test): 2025-04-08-аас хойш, загваруудыг ОГТ өөрчлөхгүйгээр.

Бодит цагийн загварууд (models/*.txt) 2025-04-06 хүртэлх өгөгдлөөр сургагдсан. Энэ скрипт тэдгээрийг
календарь, ханшийг сунгасны дараах хугацаанд үнэлнэ. Дипломын хамгийн хүчтэй нотолгоо нь энэ хэсэг:
загвар, параметр, дүрмийг энэ өгөгдлийг харахаас өмнө тогтоосон.

1. Хувилбарын загвар (live_news = A+S): log-loss vs сургалтын давтамж, чиглэл/хэмжээний AUC
2. 24 цагийн хүрээний загвар (vol_news): R², 80% интервалын хамрах хувь, HAR-аас сайжралт
3. Мэдээ (M5): surprise -> эхний 5 минутын чиглэлийн таарц; 60 минутын хүрээг төрлийн профайлаар (2025-04 хүртэлх) таамаглах
Харьцуулалтад test (2023-01 – 2025-04-07)-ийн тоог reports/metrics.json, vol_metrics.json-оос авна.

Өмнө нь: merge_calendar.py -> run_pipeline.py --end 2026-12-01 -> download_m1_dukascopy.py
Ажиллуулах: python scripts/eval_forward.py
  (--start 2024-10-01: зөвхөн кодыг шалгах зорилгоор test хугацааны хэсэг дээр ажиллуулах)
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lightgbm as lgb  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from fxdata import calendar  # noqa: E402
from fxmodel.features import build_features, group_columns  # noqa: E402
from fxmodel.intraday import future_range_bars, load_m5_cached, seasonal_baseline  # noqa: E402
from fxmodel.news_events import apply_type_effect, build_events, type_effect, usual_effect  # noqa: E402
from fxmodel.scenarios import future_paths  # noqa: E402
from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
from fxmodel.volatility import future_range, har_features  # noqa: E402

REP, MOD = ROOT / "reports", ROOT / "models"
HAR_COLS = ["har_d", "har_w", "har_m", "har_hour_sin", "har_hour_cos"]
RNG = np.random.default_rng(0)


def week_boot(vals: np.ndarray, weeks, n=1000):
    d = pd.Series(vals).groupby(np.asarray(weeks)).agg(["sum", "count"])
    out = []
    for _ in range(n):
        s = d.sample(len(d), replace=True, random_state=RNG.integers(1 << 31))
        out.append(s["sum"].sum() / s["count"].sum())
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def wilson(k, n):
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return dict(n=int(n), rate=round(100 * k / n, 1) if n else None, ci95=[round(100 * lo, 1), round(100 * hi, 1)])


def scenarios(df, start):
    meta = json.loads((MOD / "meta.json").read_text(encoding="utf-8"))
    with open(MOD / "scenario_model.pkl", "rb") as fh:
        sm = pickle.load(fh)
    info = meta["models"]["live_news"]
    X = build_features(df)
    paths = future_paths(df, meta["horizon"])
    g = group_columns(df)
    ok = paths.notna().all(axis=1) & X[g["A"]].notna().all(axis=1) & X["hours_to_next_event"].notna()
    fw = ok & (df.index >= start)
    if fw.sum() < 100:
        return {"n": int(fw.sum()), "note": "өгөгдөл хэт цөөн"}
    y = sm.assign(paths[fw])
    p = lgb.Booster(model_file=str(MOD / "live_news.txt")).predict(X.loc[fw, info["columns"]])
    base = np.tile(sm.base_rates, (len(y), 1))
    ll_m = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    ll_b = -np.log(np.clip(base[np.arange(len(y)), y], 1e-12, 1))
    weeks = df.index[fw].tz_localize(None).to_period("W").astype(str)
    end = sm.centroids[:, -1]
    move = np.abs(sm.centroids).max(axis=1)
    pf = paths[fw]
    big = pf.abs().max(axis=1) > pf.abs().max(axis=1).median()
    test = json.loads((REP / "metrics.json").read_text(encoding="utf-8"))
    return dict(n=int(fw.sum()), period=[str(df.index[fw].min()), str(df.index[fw].max())],
                logloss=round(float(ll_m.mean()), 4), baseline_logloss=round(float(ll_b.mean()), 4),
                gain_pct=round(100 * (1 - ll_m.mean() / ll_b.mean()), 2),
                gain_ci95_pct=[round(100 * v / ll_b.mean(), 2) for v in week_boot(ll_b - ll_m, weeks)],
                direction_auc=round(float(roc_auc_score(pf["k24"] > 0, p[:, end > 0].sum(1))), 3),
                magnitude_auc=round(float(roc_auc_score(big, p[:, move > np.median(move)].sum(1))), 3),
                realized_freq={n: round(float((y == i).mean()), 3) for i, n in enumerate(sm.names)},
                test_reference=dict(gain_pct=round(test["A+D(хуваарь)"]["logloss_gain_vs_baseline_pct"], 2),
                                    direction_auc=round(test["direction_auc"], 3), magnitude_auc=round(test["magnitude_auc"], 3)))


def volatility(df, start):
    vm = json.loads((MOD / "vol_meta.json").read_text(encoding="utf-8"))["models"]["vol_news"]
    X = build_features(df).join(har_features(df))
    yp = future_range(df, 24)
    y = np.log(yp.clip(lower=1))
    g = group_columns(df)
    ok = yp.notna() & X[HAR_COLS + g["A"]].notna().all(axis=1) & X["hours_to_next_event"].notna()
    idx = df.index
    tr = ok & (idx < TRAIN_END - pd.Timedelta(hours=24))
    va = ok & (idx >= TRAIN_END) & (idx < VALID_END - pd.Timedelta(hours=24))
    fw = ok & (idx >= start)
    A = np.c_[np.ones(tr.sum()), X.loc[tr, HAR_COLS].to_numpy()]
    beta, *_ = np.linalg.lstsq(A, y[tr].to_numpy(), rcond=None)
    har = lambda m: np.c_[np.ones(m.sum()), X.loc[m, HAR_COLS].to_numpy()] @ beta  # noqa: E731
    q_lo, q_hi = np.quantile(y[va].to_numpy() - har(va), [0.1, 0.9])
    pred = {a: lgb.Booster(model_file=str(MOD / i["file"])).predict(X.loc[fw, i["columns"]]) for a, i in vm.items()}
    yt = y[fw].to_numpy()
    h = har(fw)
    ss = np.sum((yt - yt.mean()) ** 2)
    weeks = idx[fw].tz_localize(None).to_period("W").astype(str)
    test = json.loads((REP / "vol_metrics.json").read_text(encoding="utf-8"))
    return dict(n=int(fw.sum()), median_range_pips=round(float(np.median(np.exp(yt))), 1),
                r2_lgbm=round(float(1 - np.sum((yt - pred["0.5"]) ** 2) / ss), 3),
                r2_har=round(float(1 - np.sum((yt - h) ** 2) / ss), 3),
                mae_pips_lgbm=round(float(np.mean(np.abs(np.exp(yt) - np.exp(pred["0.5"])))), 1),
                mae_pips_har=round(float(np.mean(np.abs(np.exp(yt) - np.exp(h)))), 1),
                mae_gain_vs_har_ci95=week_boot(np.abs(yt - h) - np.abs(yt - pred["0.5"]), weeks),
                coverage80_lgbm=round(float(np.mean((yt >= pred["0.1"]) & (yt <= pred["0.9"]))), 3),
                coverage80_har=round(float(np.mean((yt >= h + q_lo) & (yt <= h + q_hi))), 3),
                test_reference=dict(r2_lgbm=round(test["LGBM+S"]["r2_log"], 3), r2_har=round(test["HAR"]["r2_log"], 3),
                                    coverage80=round(test["LGBM+S"]["coverage80"], 3))), (idx[fw], yt, pred, h)


def news(start):
    m5 = load_m5_cached(ROOT)
    raw = ROOT / "data/raw/calendar/forex_factory.csv"
    ev, _ = calendar.load_calendar(raw, "Asia/Tehran")
    E = build_events(m5, ev, usual_effect(raw))
    seas = seasonal_baseline(m5, np.log(future_range_bars(m5, 12).clip(lower=0.5)))
    E["seasonal"] = seas.to_numpy()[E["p"].to_numpy() - 1]
    E = E[E["seasonal"].notna() & (E["R60"] > 0)].copy()
    E["y"] = np.log(E["R60"])
    past, fw = E[E.index < min(start, TEST_END)], E[E.index >= start]
    if len(fw) < 10:
        return {"n_events": int(len(fw)), "note": "мэдээ хэт цөөн (календарь сунгагдаагүй?)"}
    tab, glob = type_effect(past.assign(excess=past["y"] - past["seasonal"]), "excess")
    pred = fw["seasonal"] + apply_type_effect(fw, tab, glob)
    resid = past["y"] - (past["seasonal"] + apply_type_effect(past, tab, glob))
    lo, hi = np.quantile(resid, [0.1, 0.9])
    ss = np.sum((fw["y"] - fw["y"].mean()) ** 2)
    out = dict(n_events=int(len(fw)), period=[str(fw.index.min()), str(fw.index.max())],
               r60_median_pips=round(float(fw["R60"].median()), 1),
               r2_seasonal_only=round(float(1 - np.sum((fw["y"] - fw["seasonal"]) ** 2) / ss), 3),
               r2_seasonal_plus_type=round(float(1 - np.sum((fw["y"] - pred) ** 2) / ss), 3),
               mae_pips_type=round(float(np.mean(np.abs(fw["R60"] - np.exp(pred)))), 1),
               coverage80=round(float(np.mean((fw["y"] >= pred + lo) & (fw["y"] <= pred + hi))), 3))
    D = fw[(fw["S_max"].abs() >= 0.5) & (fw["J"] != 0)]
    hit = (np.sign(D["J"]) == np.sign(D["S_max"]))
    conflict = (D["n_pos"] > 0) & (D["n_neg"] > 0)
    out["surprise_to_first5m"] = dict(all=wilson(int(hit.sum()), len(D)),
                                      no_conflict=wilson(int(hit[~conflict].sum()), int((~conflict).sum())),
                                      test_reference_pct=72.9)
    by_t = pd.DataFrame({"t": D["key_title"], "hit": hit}).groupby("t")["hit"].agg(["mean", "count"])
    out["by_type_n>=5"] = {t: dict(rate=round(100 * r["mean"], 1), n=int(r["count"]))
                           for t, r in by_t[by_t["count"] >= 5].sort_values("mean", ascending=False).iterrows()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=str(TEST_END.date()), help="Forward эхлэх огноо (анхдагч 2025-04-08)")
    a = ap.parse_args()
    start = pd.Timestamp(a.start, tz="UTC")
    dry = start < TEST_END
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    print(f"H1 өгөгдөл: {df.index.min()} – {df.index.max()}")
    if df.index.max() < start + pd.Timedelta(days=30):
        sys.exit(f"{start.date()}-аас хойш өгөгдөл хангалтгүй. Календарь, ханшаа сунгаад run_pipeline.py --end 2026-12-01 ажиллуулна уу.")
    res = {"start": str(start), "dry_run_on_test_period": dry}
    res["scenario_model"] = scenarios(df, start)
    res["range_model_24h"], (t, yt, pred, h) = volatility(df, start)
    res["news"] = news(start)
    name = "forward_dryrun" if dry else "forward"
    (REP / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 4))
    s = pd.DataFrame({"real": np.exp(yt), "q50": np.exp(pred["0.5"]), "q10": np.exp(pred["0.1"]),
                      "q90": np.exp(pred["0.9"]), "har": np.exp(h)}, index=t).resample("D").last().dropna()
    ax.fill_between(s.index, s.q10, s.q90, color="#4f8cf0", alpha=0.2, label="Загвар 80% интервал")
    ax.plot(s.index, s.q50, color="#4f8cf0", lw=1.2, label="Загвар (медиан)")
    ax.plot(s.index, s.har, color="#999", lw=1, ls="--", label="HAR")
    ax.plot(s.index, s.real, color="black", lw=1, label="Бодит")
    r = res["range_model_24h"]
    ax.set_title(f"Forward {start:%Y-%m-%d}–: ирэх 24 цагийн хүрээ (өдөр бүрийн сүүлийн таамаг). "
                 f"R² {r['r2_lgbm']:.2f} vs HAR {r['r2_har']:.2f}, хамрах {100 * r['coverage80_lgbm']:.0f}%", fontsize=10)
    ax.set_ylabel("пип")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(REP / f"{name}_range.png", dpi=130)
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
