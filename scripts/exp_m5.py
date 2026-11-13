"""Туршилт: 5 минутын өгөгдөл дээр ирэх 1 цагийн хүрээг олон timeframe-ийн шинжээр таамаглах.

Өгөгдөл: data/raw/histdata/ (HistData M1, 2015–2025) -> M5. Календарийн хүрээнд (2025-04-07 хүртэл).
Хуваалт: train 2015–2021, valid 2022, test 2023–2025-04 (1 цагийн purge).
Baseline:
  persistence : сүүлийн 1 цагийн хүрээ
  seasonal    : өдрийн ижил цагийн сүүлийн 20 өдрийн медиан
  HAR-ID      : OLS(1ц, 4ц, 1 өдөр, 5 өдөр, seasonal, өдрийн цаг)
LightGBM quantile (10/50/90):
  L           : M5 суурь + HAR + seasonal
  L+MTF       : + H1/H4 нөхцөл (бүтэц, premium/discount, хэлбэлзэл)
  L+MTF+S     : + мэдээний хуваарь минутаар
Нэмэлт шалгалт: ижил шинжээр ирэх 1 цагийн ЧИГЛЭЛИЙН AUC.

Ажиллуулах: python scripts/exp_m5.py      (~5–10 мин)
"""
import json
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
from sklearn.metrics import roc_auc_score  # noqa: E402

from fxdata import calendar  # noqa: E402
from fxmodel.intraday import load_m5_cached  # noqa: E402
from fxmodel.intraday import (future_range_bars, htf_context, load_m5_from_histdata, m5_base,  # noqa: E402
                              news_minutes, seasonal_baseline)
from fxmodel.volatility import pinball  # noqa: E402

N = 12                                         # 12 × 5мин = 1 цаг
from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
REP = ROOT / "reports"
QS = (0.1, 0.5, 0.9)
PARAMS = dict(objective="quantile", learning_rate=0.05, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=0, num_threads=8)
STRIDE = 3                                     # train/valid-д 3 лаа тутмын нэгийг авна (давхардсан шошго, хурд)


def bootstrap_mae_diff(y, a, b, groups, n=300, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.Series(np.abs(y - a) - np.abs(y - b)).groupby(groups).agg(["sum", "count"])
    out = []
    for _ in range(n):
        s = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31))
        out.append(s["sum"].sum() / s["count"].sum())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    REP.mkdir(exist_ok=True)
    m5 = load_m5_cached(ROOT)
    ev, _ = calendar.load_calendar(ROOT / "data/raw/calendar/forex_factory.csv", "Asia/Tehran")
    m5 = m5[m5.index < ev["ts_utc"].max().floor("D")]
    print("M5:", len(m5), m5.index.min(), m5.index.max())

    y_pips = future_range_bars(m5, N)
    y = np.log(y_pips.clip(lower=0.5))
    base = m5_base(m5)
    base["seasonal"] = seasonal_baseline(m5, y)
    htf = htf_context(m5)
    news = news_minutes(m5, ev)
    X = base.join(htf).join(news)
    har_cols = ["har_1h", "har_4h", "har_1d", "har_5d", "seasonal", "tod_sin", "tod_cos"]
    sets = {"L": list(base.columns), "L+MTF": list(base.columns) + list(htf.columns),
            "L+MTF+S": list(base.columns) + list(htf.columns) + list(news.columns)}

    idx = m5.index
    ok = y.notna() & X[har_cols].notna().all(axis=1) & (m5["n_m1"] >= 4)
    purge = pd.Timedelta(hours=1)
    pos = np.arange(len(idx))
    tr = ok & (idx < TRAIN_END - purge) & (pos % STRIDE == 0)
    va = ok & (idx >= TRAIN_END) & (idx < VALID_END - purge) & (pos % STRIDE == 0)
    te = ok & (idx >= VALID_END) & (idx < TEST_END - purge)
    yte = y[te].to_numpy()
    days = idx[te].tz_localize(None).normalize().astype(str)
    print("train/valid/test:", int(tr.sum()), int(va.sum()), int(te.sum()))

    pred, res = {}, {}
    pred["persistence"] = {0.5: X.loc[te, "har_1h"].to_numpy()}
    pred["seasonal"] = {0.5: X.loc[te, "seasonal"].to_numpy()}
    A = lambda d: np.c_[np.ones(len(d)), d[har_cols].to_numpy()]  # noqa: E731
    beta, *_ = np.linalg.lstsq(A(X[tr]), y[tr].to_numpy(), rcond=None)
    resid = y[va].to_numpy() - A(X[va]) @ beta
    lo, hi = np.quantile(resid, [0.1, 0.9])
    h = A(X[te]) @ beta
    pred["HAR-ID"] = {0.1: h + lo, 0.5: h, 0.9: h + hi}

    for name, cols in sets.items():
        pred[name] = {}
        for a in QS:
            m = lgb.train(PARAMS | {"alpha": a}, lgb.Dataset(X.loc[tr, cols], y[tr]), 2000,
                          valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            pred[name][a] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
            if name == "L+MTF+S" and a == 0.5:
                imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
        print("trained", name)

    ss = np.sum((yte - yte.mean()) ** 2)
    for name, p in pred.items():
        q = p[0.5]
        r = dict(mae_log=float(np.mean(np.abs(yte - q))), r2_log=float(1 - np.sum((yte - q) ** 2) / ss),
                 mae_pips=float(np.mean(np.abs(np.exp(yte) - np.exp(q)))))
        if 0.1 in p:
            r["coverage80"] = float(np.mean((yte >= p[0.1]) & (yte <= p[0.9])))
            r["pinball_avg"] = float(np.mean([pinball(yte, p[a], a) for a in QS]))
        if name != "HAR-ID":
            r["mae_vs_HAR_ci95"] = bootstrap_mae_diff(yte, pred["HAR-ID"][0.5], q, days)
        res[name] = r
    res["L+MTF"]["mae_vs_L_ci95"] = bootstrap_mae_diff(yte, pred["L"][0.5], pred["L+MTF"][0.5], days)
    res["L+MTF+S"]["mae_vs_L+MTF_ci95"] = bootstrap_mae_diff(yte, pred["L+MTF"][0.5], pred["L+MTF+S"][0.5], days)

    soon = (X.loc[te, "min_to_next_event"] <= 60).to_numpy()
    res["by_news_next_60m"] = {n: dict(mae_with_news=float(np.mean(np.abs(yte[soon] - pred[n][0.5][soon]))),
                                       mae_no_news=float(np.mean(np.abs(yte[~soon] - pred[n][0.5][~soon]))))
                               for n in ("HAR-ID", "L+MTF", "L+MTF+S")}
    res["n_news_next_60m"] = int(soon.sum())
    res["test_median_range_pips"] = float(np.median(np.exp(yte)))
    res["top_features_L+MTF+S"] = (100 * imp / imp.sum()).round(1).head(15).to_dict()

    # ---- Чиглэлийн шалгалт: ирэх 1 цагийн өгөөжийн тэмдэг ----
    c = m5["close"]
    fut = (c.shift(-N) - c)
    yd = (fut > 0).astype(int)
    okd = ok & fut.notna() & (fut != 0)
    cols = sets["L+MTF+S"]
    clf = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=1000,
                         feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, verbose=-1, seed=0, num_threads=8),
                    lgb.Dataset(X.loc[tr & okd, cols], yd[tr & okd]), 1000,
                    valid_sets=[lgb.Dataset(X.loc[va & okd, cols], yd[va & okd])],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    pdir = clf.predict(X.loc[te & okd, cols], num_iteration=clf.best_iteration)
    res["direction_auc_next_1h"] = float(roc_auc_score(yd[te & okd], pdir))
    res["direction_best_iter"] = int(clf.best_iteration)
    res["split"] = dict(train=int(tr.sum()), valid=int(va.sum()), test=int(te.sum()))
    (REP / "m5_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---- График: нэг долоо хоног, бодит vs таамаг ----
    s = pd.DataFrame({"real": np.exp(yte), "q10": np.exp(pred["L+MTF+S"][0.1]), "q50": np.exp(pred["L+MTF+S"][0.5]),
                      "q90": np.exp(pred["L+MTF+S"][0.9]), "har": np.exp(pred["HAR-ID"][0.5])}, index=idx[te])
    w = s["2024-03-04":"2024-03-09"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(w.index, w.q10, w.q90, alpha=0.25, label="L+MTF+S 80% интервал")
    ax.plot(w.index, w.q50, lw=1.4, label="L+MTF+S медиан")
    ax.plot(w.index, w.har, lw=1, ls="--", color="grey", label="HAR-ID")
    ax.plot(w.index, w.real, lw=0.8, color="black", label="Бодит 1ц хүрээ")
    ax.set_ylabel("Ирэх 1 цагийн хүрээ, пип")
    ax.set_title("M5: ирэх 1 цагийн хүрээний таамаглал (test, 2024-03-04 – 03-08)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(REP / "m5_forecast.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
