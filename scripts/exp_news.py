"""Мэдээний үеийн шинжилгээ: (1) хариу үйлдлийн хэмжээ, (2) surprise -> эхний 5 мин, (3) дараагийн 60 мин.

Өгөгдөл: M5 (HistData), өндөр нөлөөтэй USD/EUR мэдээ 2015 – 2025-04. Хуваалт: train 2015–2021, valid 2022, test 2023+.

(1) Мэдээ гарахаас ӨМНӨ: дараагийн 60 минутын хүрээ (log пип).
    B0 улирал       : өдрийн ижил цагийн ердийн хүрээ (мэдээний мэдээлэлгүй)
    B1 HAR          : OLS(сүүлийн 1ц, 24ц хүрээ, улирал)
    B2 HAR+төрөл    : + мэдээний төрлийн түүхэн нөлөө (train-ээс)
    LGBM            : бүх шинж, quantile 10/50/90
(2) Мэдээ гарах мөчид: эхний 5 минутын чиглэл surprise-ийн чиглэлтэй таарах уу (өгөгдлийн зөв эсэхийн шалгалт).
(3) Эхний 5 минутын ДАРАА: дараагийн 60 минутын чиглэл. Дүрмүүд (үргэлжлүүлэх / эсрэг / surprise-ийг дагах)
    ба LightGBM. Арилжааны туршилт: p+1 лааны нээлтэд орж p+12-ийн хаалтад гарна, зардал 2 пип (мэдээний spread).

Ажиллуулах: python scripts/exp_news.py
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
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from fxdata import calendar  # noqa: E402
from fxmodel.intraday import load_m5_cached  # noqa: E402
from fxmodel.intraday import future_range_bars, load_m5_from_histdata, seasonal_baseline  # noqa: E402
from fxmodel.news_events import apply_type_effect, build_events, type_effect, usual_effect  # noqa: E402

from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
REP = ROOT / "reports"
COST = 2.0


def ols(Xtr, ytr):
    A = lambda X: np.c_[np.ones(len(X)), X]  # noqa: E731
    beta, *_ = np.linalg.lstsq(A(Xtr), ytr, rcond=None)
    return lambda X: A(X) @ beta


def boot_ci(diff, groups, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.Series(diff).groupby(groups).agg(["sum", "count"])
    out = [(lambda s: s["sum"].sum() / s["count"].sum())(d.sample(len(d), replace=True, random_state=rng.integers(1 << 31)))
           for _ in range(n)]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    m5 = load_m5_cached(ROOT)
    raw = ROOT / "data/raw/calendar/forex_factory.csv"
    ev, _ = calendar.load_calendar(raw, "Asia/Tehran")
    E = build_events(m5, ev, usual_effect(raw))
    seas = seasonal_baseline(m5, np.log(future_range_bars(m5, 12).clip(lower=0.5)))
    E["seasonal"] = seas.to_numpy()[E["p"].to_numpy() - 1]          # p-1 лааны "ирэх 1 цаг" = мэдээний цаг
    E = E[E["seasonal"].notna() & (E["R60"] > 0)].copy()
    E["y"] = np.log(E["R60"])
    E["l_pre60"], E["l_pre24"] = np.log(E["pre_r60"].clip(lower=0.5)), np.log(E["pre_r24h"].clip(lower=0.5))
    tr, va, te = E.index < TRAIN_END, (E.index >= TRAIN_END) & (E.index < VALID_END), (E.index >= VALID_END) & (E.index < TEST_END)
    tab, glob = type_effect(E[tr].assign(excess=E["y"] - E["seasonal"]), "excess")
    E["type_eff"] = apply_type_effect(E, tab, glob)
    tabz, globz = type_effect(E[tr].assign(absS=E["S_max"].abs().fillna(0)), "absS")
    E["type_abs_surprise"] = apply_type_effect(E, tabz, globz)
    days = E.index[te].tz_localize(None).normalize().astype(str)
    res = {"n_events": {"train": int(tr.sum()), "valid": int(va.sum()), "test": int(te.sum())}}

    # ---------------- (1) Хариу үйлдлийн хэмжээ ----------------
    y = E["y"].to_numpy()
    preds = {"B0 улирал": E["seasonal"].to_numpy()[te]}
    f1 = ols(E.loc[tr, ["l_pre60", "l_pre24", "seasonal"]].to_numpy(), y[tr])
    preds["B1 HAR"] = f1(E.loc[te, ["l_pre60", "l_pre24", "seasonal"]].to_numpy())
    cols2 = ["l_pre60", "l_pre24", "seasonal", "type_eff"]
    f2 = ols(E.loc[tr, cols2].to_numpy(), y[tr])
    preds["B2 HAR+төрөл"] = f2(E.loc[te, cols2].to_numpy())
    feats = cols2 + ["n_events", "n_usd", "n_eur", "hour", "dow", "type_abs_surprise"]
    q = {}
    for a in (0.1, 0.5, 0.9):
        m = lgb.train(dict(objective="quantile", alpha=a, learning_rate=0.03, num_leaves=7, min_data_in_leaf=40,
                           feature_fraction=0.9, lambda_l2=5.0, verbose=-1, seed=0, num_threads=2),
                      lgb.Dataset(E.loc[tr, feats], y[tr]), 2000, valid_sets=[lgb.Dataset(E.loc[va, feats], y[va])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        q[a] = m.predict(E.loc[te, feats], num_iteration=m.best_iteration)
    preds["LGBM"] = q[0.5]
    yt = y[te]
    ss = np.sum((yt - yt.mean()) ** 2)
    r1 = {}
    for k, p in preds.items():
        r1[k] = dict(mae_log=float(np.mean(np.abs(yt - p))), r2=float(1 - np.sum((yt - p) ** 2) / ss),
                     mae_pips=float(np.mean(np.abs(np.exp(yt) - np.exp(p)))))
        if k != "B1 HAR":
            r1[k]["vs_B1_ci95"] = boot_ci(np.abs(yt - preds["B1 HAR"]) - np.abs(yt - p), days)
    r1["LGBM"]["coverage80"] = float(np.mean((yt >= q[0.1]) & (yt <= q[0.9])))
    r1["test_median_R60_pips"] = float(np.median(np.exp(yt)))
    top = pd.Series(tab).sort_values(ascending=False)
    r1["type_effect_top"] = {k: round(float(np.exp(v)), 2) for k, v in top.head(10).items()}
    r1["type_effect_bottom"] = {k: round(float(np.exp(v)), 2) for k, v in top.tail(5).items()}
    res["1_reaction_size"] = r1

    # ---------------- (2) Surprise -> эхний 5 минут ----------------
    s2 = E[E["S_max"].abs() >= 0.5]
    r2 = {}
    for name, msk in (("бүгд", np.ones(len(s2), bool)), ("train", s2.index < TRAIN_END), ("test", (s2.index >= VALID_END) & (s2.index < TEST_END))):
        x = s2[msk]
        x = x[x["J"] != 0]
        hit = np.sign(x["J"]) == np.sign(x["S_max"])
        r2[name] = dict(n=int(len(x)), hit_rate=float(hit.mean()), p=float(stats.binomtest(int(hit.sum()), len(x)).pvalue),
                        corr_J_S=float(np.corrcoef(x["J"], x["S_max"])[0, 1]))
    big = s2[(s2["S_max"].abs() >= 1.5) & (s2["J"] != 0)]
    r2["|S|>=1.5"] = dict(n=int(len(big)), hit_rate=float((np.sign(big["J"]) == np.sign(big["S_max"])).mean()),
                          median_abs_J_pips=float(big["J"].abs().median()))
    res["2_surprise_to_jump"] = r2

    # ---------------- (3) Эхний 5 минутын дараах 60 минут ----------------
    D = E[(E["J"] != 0) & (E["POST"] != 0)].copy()
    D["up"] = (D["POST"] > 0).astype(int)
    D["J_rel"] = D["J"] / D["pre_r24h"].clip(lower=1)
    D["J_vs_exp"] = D["J"].abs() / np.exp(D["seasonal"] + D["type_eff"])
    D["S0"] = D["S_max"].fillna(0)
    D["J_dir_S"] = np.sign(D["J"]) * np.sign(D["S0"])
    f3 = ["J", "J_rel", "J_vs_exp", "S0", "J_dir_S", "type_eff", "l_pre60", "l_pre24", "hour", "n_events"]
    tr3, va3, te3 = D.index < TRAIN_END, (D.index >= TRAIN_END) & (D.index < VALID_END), (D.index >= VALID_END) & (D.index < TEST_END)
    clf = lgb.train(dict(objective="binary", learning_rate=0.02, num_leaves=7, min_data_in_leaf=60, feature_fraction=0.9,
                         lambda_l2=5.0, verbose=-1, seed=0, num_threads=2),
                    lgb.Dataset(D.loc[tr3, f3], D.loc[tr3, "up"]), 2000,
                    valid_sets=[lgb.Dataset(D.loc[va3, f3], D.loc[va3, "up"])], callbacks=[lgb.early_stopping(100, verbose=False)])
    pu = clf.predict(D.loc[te3, f3], num_iteration=clf.best_iteration)
    T = D[te3]
    rules = {"үргэлжлүүлэх (J-ийн дагуу)": np.sign(T["J"]), "эсрэг (J-ийн эсрэг)": -np.sign(T["J"]),
             "surprise-ийг дагах": np.sign(T["S0"]).replace(0, np.nan), "LightGBM": np.where(pu > 0.5, 1, -1)}
    r3 = {"n_test": int(len(T)), "train_continuation_rate": float((np.sign(D.loc[tr3, "J"]) == np.sign(D.loc[tr3, "POST"])).mean()),
          "test_continuation_rate": float((np.sign(T["J"]) == np.sign(T["POST"])).mean()),
          "lgbm_auc": float(roc_auc_score(T["up"], pu)), "lgbm_best_iter": int(clf.best_iteration)}
    tdays = T.index.tz_localize(None).normalize().astype(str)
    for name, d in rules.items():
        d = pd.Series(np.asarray(d, float), index=T.index)
        ok = d.notna() & (d != 0)
        pnl = d[ok] * T.loc[ok, "POST"] - COST
        r3[name] = dict(trades=int(ok.sum()), hit=float((d[ok] == np.sign(T.loc[ok, "POST"])).mean()),
                        avg_pips=float(pnl.mean()), total_pips=float(pnl.sum()),
                        avg_pips_ci95=boot_ci(pnl.to_numpy(), tdays[ok.to_numpy()]))
    res["3_post_release_60m"] = r3
    (REP / "exp_news.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    E.to_csv(REP / "news_events.csv")
    print(json.dumps(res, ensure_ascii=False, indent=1))

    # ---------------- Графикууд ----------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x = s2[s2["J"] != 0]
    axes[0].scatter(x["S_max"], x["J"], s=6, alpha=0.35)
    axes[0].axhline(0, color="grey", lw=0.6)
    axes[0].axvline(0, color="grey", lw=0.6)
    axes[0].set_xlabel("Surprise (z, EUR/USD-д нөлөөлөх чиглэлээр)")
    axes[0].set_ylabel("Эхний 5 минутын хөдөлгөөн, пип")
    axes[0].set_title(f"Surprise → эхний 5 мин: чиглэл таарсан {100 * r2['бүгд']['hit_rate']:.0f}%")
    pt = q[0.5]
    axes[1].scatter(np.exp(pt), np.exp(yt), s=8, alpha=0.4)
    lim = [5, max(np.exp(yt).max(), 150)]
    axes[1].plot(lim, lim, "k--", lw=0.8)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Таамагласан 60 минутын хүрээ, пип (мэдээ гарахаас өмнө)")
    axes[1].set_ylabel("Бодит, пип")
    axes[1].set_title(f"Хариу үйлдлийн хэмжээ (test): R² {r1['LGBM']['r2']:.2f}")
    fig.tight_layout()
    fig.savefig(REP / "news_analysis.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
