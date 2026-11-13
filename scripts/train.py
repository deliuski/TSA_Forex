"""Хувилбарын загварыг сургаж, шударга үнэлгээ хийнэ.

Хуваалт (цагаар, давхардалгүй):
  train : 2015-01 .. 2021-12   (k-means хувилбарууд ба LightGBM)
  valid : 2022                 (early stopping)
  test  : 2023-01 .. 2025-04   (нэг ч удаа тааруулахад ашиглаагүй)
Хил бүрт 24 цагийн "purge": шошго 24 цаг урагш хардаг тул train-ийн сүүлийн
шошго valid-ийн өгөгдөлд хүрэхгүй байх ёстой.

Харьцуулалт (ablation): baseline (түүхэн давтамж) | A | A+B | A+C | A+D | A+B+C+D
Үр дүн: reports/ (графикууд, metrics.json), models/ (бодит цагт ашиглах загвар).

Ажиллуулах: python scripts/train.py
"""
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

from fxmodel.cross_asset import cross_features, load_markets
from fxmodel.features import NEWS_SCHEDULE_COLS, build_features, group_columns  # noqa: E402
from fxmodel.scenarios import ScenarioModel, future_paths  # noqa: E402

H = 24
from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
PARAMS = dict(objective="multiclass", learning_rate=0.03, num_leaves=31,
              min_data_in_leaf=300, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=0, num_threads=2)
SETS = {"A": "A", "A+B": "AB", "A+C": "AC", "A+D": "AD", "A+B+C+D": "ABCD", "A+D(хуваарь)": "AS"}
# Бодит цагт ашиглах загварууд: мэдээний хуваарьтай ба мэдээгүй (календарь татагдахгүй үед)
LIVE_SETS = {"live_news": "A+D(хуваарь)", "live_nonews": "A"}
REP, MOD = ROOT / "reports", ROOT / "models"


def brier(y, p):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(axis=1).mean())


def block_bootstrap_diff(y, p_a, p_b, weeks, n=500, seed=0):
    """Долоо хоногоор блок bootstrap: logloss(a) - logloss(b)-ийн 95% интервал."""
    rng = np.random.default_rng(seed)
    la = -np.log(np.clip(p_a[np.arange(len(y)), y], 1e-12, 1))
    lb = -np.log(np.clip(p_b[np.arange(len(y)), y], 1e-12, 1))
    d = pd.Series(la - lb).groupby(weeks).agg(["sum", "count"])
    out = []
    for _ in range(n):
        s = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31))
        out.append(s["sum"].sum() / s["count"].sum())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    REP.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    X_all = build_features(df)
    paths = future_paths(df, H)
    groups = group_columns(df)
    groups["S"] = NEWS_SCHEDULE_COLS
    # Бусад зах зээл (scripts/download_markets.py-аар татсан бол)
    mk = load_markets(ROOT / "data/raw/markets")
    if mk:
        xf = cross_features(df, mk)
        X_all = X_all.join(xf)
        groups["X"] = list(xf.columns)
        SETS["A+X"] = "AX"
        SETS["A+S+X"] = "ASX"
        print("Бусад зах зээл:", sorted(mk), f"({len(xf.columns)} шинж)")
    ok = paths.notna().all(axis=1) & X_all[groups["A"] + groups["B"] + groups["C"]].notna().all(axis=1)
    ok &= df.index >= df.index[300]
    K = None  # ScenarioModel-оос

    purge = pd.Timedelta(hours=H)
    idx = df.index
    tr = ok & (idx < TRAIN_END - purge)
    va = ok & (idx >= TRAIN_END) & (idx < VALID_END - purge)
    te = ok & (idx >= VALID_END) & (idx < TEST_END - purge)

    sm = ScenarioModel().fit(paths[tr])
    K = sm.k
    PARAMS["num_class"] = K
    y = pd.Series(-1, index=idx)
    y[ok] = sm.assign(paths[ok])
    print("Хувилбарууд:", list(zip(sm.names, sm.base_rates.round(3))))

    results, probs, models = {}, {}, {}
    yte = y[te].to_numpy()
    base_p = np.tile(sm.base_rates, (te.sum(), 1))
    probs["baseline"] = base_p
    results["baseline"] = dict(logloss=log_loss(yte, base_p, labels=range(K)), brier=brier(yte, base_p),
                               acc=float((base_p.argmax(1) == yte).mean()))
    # Хоёр дахь (илүү хүчтэй байж болох) baseline: сүүлийн ~3 сарын давтамж (24ц purge-тэй)
    oh = pd.DataFrame(np.eye(K)[y.clip(lower=0)], index=idx).where(ok, np.nan)
    roll = oh.shift(H).rolling(24 * 65, min_periods=500).mean()[te].to_numpy() + 1e-3
    roll /= roll.sum(axis=1, keepdims=True)
    results["baseline_rolling3m"] = dict(logloss=log_loss(yte, roll, labels=range(K)), brier=brier(yte, roll),
                                         acc=float((roll.argmax(1) == yte).mean()))
    for name, g in SETS.items():
        cols = sum((groups[c] for c in g), [])
        dtr = lgb.Dataset(X_all.loc[tr, cols], y[tr])
        dva = lgb.Dataset(X_all.loc[va, cols], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, 3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        p = m.predict(X_all.loc[te, cols], num_iteration=m.best_iteration)
        probs[name], models[name] = p, (m, cols)
        top2 = np.argsort(-p, axis=1)[:, :2]
        results[name] = dict(logloss=log_loss(yte, p, labels=range(K)), brier=brier(yte, p),
                             acc=float((p.argmax(1) == yte).mean()),
                             top2_acc=float((top2 == yte[:, None]).any(axis=1).mean()),
                             best_iter=int(m.best_iteration), n_features=len(cols))
        print(name, {k: round(v, 4) for k, v in results[name].items()})

    weeks = idx[te].tz_localize(None).to_period("W").astype(str)
    for name in SETS:
        r = results[name]
        r["logloss_gain_vs_baseline_pct"] = 100 * (1 - r["logloss"] / results["baseline"]["logloss"])
        r["gain_ci95"] = block_bootstrap_diff(yte, base_p, probs[name], weeks)   # >0 = baseline-аас сайн
    for name in [n for n in SETS if n != "A"]:
        results[name]["vs_A_ci95"] = block_bootstrap_diff(yte, probs["A"], probs[name], weeks)

    # Хаана ажиллаж байна: чиглэл vs хэмжээ, мэдээтэй vs мэдээгүй цаг
    full = probs["A+B+C+D"]
    end = sm.centroids[:, -1]
    p_up = full[:, end > 0].sum(1)
    move = np.abs(sm.centroids).max(axis=1)
    p_big = full[:, move > np.median(move)].sum(1)
    ptest = paths[te]
    results["direction_auc"] = float(roc_auc_score(ptest["k24"] > 0, p_up))
    big_true = ptest.abs().max(axis=1) > ptest.abs().max(axis=1).median()
    results["magnitude_auc"] = float(roc_auc_score(big_true, p_big))
    news_soon = (X_all.loc[te, "hours_to_next_event"] <= 24).to_numpy()
    for lbl, msk in (("news_within_24h", news_soon), ("no_news_24h", ~news_soon)):
        results[f"gain_{lbl}_pct"] = 100 * (1 - log_loss(yte[msk], full[msk], labels=range(K))
                                            / log_loss(yte[msk], base_p[msk], labels=range(K)))
        results[f"n_{lbl}"] = int(msk.sum())
    results["scenarios"] = [dict(id=i, name=n, base_rate=float(b), end_atr=float(c[-1]))
                            for i, (n, b, c) in enumerate(zip(sm.names, sm.base_rates, sm.centroids))]
    results["split"] = dict(train=[str(idx[tr].min()), str(idx[tr].max()), int(tr.sum())],
                            valid=[str(idx[va].min()), str(idx[va].max()), int(va.sum())],
                            test=[str(idx[te].min()), str(idx[te].max()), int(te.sum())])
    (REP / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Графикууд ----
    x = np.arange(1, H + 1)
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.6), sharex=True, sharey=True)
    for i, ax in enumerate(axes.flat):
        ax.fill_between(x, sm.q20[i], sm.q80[i], alpha=0.25)
        ax.plot(x, sm.centroids[i], lw=2)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_title(f"{i}. {sm.names[i]}  ({100 * sm.base_rates[i]:.0f}%)", fontsize=10)
    fig.supxlabel("Цаг урагш")
    fig.supylabel("Ханшийн өөрчлөлт (ATR нэгжээр)")
    fig.tight_layout()
    fig.savefig(REP / "scenarios.png", dpi=150)
    plt.close(fig)

    names = ["baseline"] + list(SETS)
    fig_names = names
    fig, ax = plt.subplots(figsize=(7, 3.8))
    gains = [0] + [results[n]["logloss_gain_vs_baseline_pct"] for n in SETS]
    err = [[0] + [100 * (results[n]["logloss_gain_vs_baseline_pct"] / 100 - results[n]["gain_ci95"][0] / results["baseline"]["logloss"]) for n in SETS],
           [0] + [100 * (results[n]["gain_ci95"][1] / results["baseline"]["logloss"] - results[n]["logloss_gain_vs_baseline_pct"] / 100) for n in SETS]]
    ax.bar(names, gains, yerr=np.abs(err), capsize=4, color=["#999"] + ["#3b6fb6"] * len(SETS))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylabel("Log-loss сайжралт, % (95% CI)")
    ax.set_title("Шинжийн бүлгүүдийн харьцуулалт (test 2023–2025)")
    fig.tight_layout()
    fig.savefig(REP / "ablation.png", dpi=150)
    plt.close(fig)

    # Калибровк: таамагласан магадлал vs бодит давтамж (бүх ангиллыг нэгтгэж)
    oh = np.eye(K)[yte]
    pp, tt = full.ravel(), oh.ravel()
    bins = np.linspace(0, 1, 11)
    which = np.clip(np.digitize(pp, bins) - 1, 0, 9)
    cal = pd.DataFrame({"p": pp, "t": tt, "b": which}).groupby("b").agg(p=("p", "mean"), t=("t", "mean"), n=("p", "size"))
    cal = cal[cal.n >= 50]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.plot(cal.p, cal.t, "o-")
    ax.set_xlabel("Таамагласан магадлал")
    ax.set_ylabel("Бодит давтамж")
    ax.set_title("Калибровк (A+B+C+D, test)")
    fig.tight_layout()
    fig.savefig(REP / "calibration.png", dpi=150)
    plt.close(fig)
    cal.to_csv(REP / "calibration.csv")

    # Бүлэг бүрийн SHAP хувь нэмэр
    m, cols = models["A+B+C+D"]
    contrib = m.predict(X_all.loc[te, cols].iloc[::5], num_iteration=m.best_iteration, pred_contrib=True)
    contrib = np.abs(contrib.reshape(len(contrib), K, len(cols) + 1)[:, :, :-1]).mean(axis=(0, 1))
    imp = pd.Series(contrib, index=cols).sort_values(ascending=False)
    imp.to_csv(REP / "shap_importance.csv")
    grp = {c: g for g, cs in groups.items() if g != "S" for c in cs}
    colors_x = {"X": "#7c6bc4"}
    results["shap_by_group_pct"] = (100 * imp.groupby(imp.index.map(grp)).sum() / imp.sum()).round(1).to_dict()
    fig, ax = plt.subplots(figsize=(7, 6))
    top = imp.head(20)[::-1]
    colors = {"A": "#888", "B": "#e3a33b", "C": "#3b6fb6", "D": "#c0504d", "X": "#7c6bc4"}
    ax.barh(top.index, top.values, color=[colors[grp[c]] for c in top.index])
    ax.set_title("Хамгийн нөлөөтэй 20 шинж (SHAP) — саарал A, шар B, цэнхэр C(ICT), улаан D(мэдээ)", fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "importance.png", dpi=150)
    plt.close(fig)
    (REP / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- Бодит цагт ашиглах загварууд: train+valid+test бүгдээр дахин сургана ----
    allm = ok & (idx < idx[te].max() - purge)
    meta = dict(horizon=H, k=K, threshold_atr=sm.s, names=sm.names, trained_until=str(idx[allm].max()), models={})
    for key, set_name in LIVE_SETS.items():
        m_set, cols = models[set_name]
        n_iter = max(int(m_set.best_iteration * 1.15), 20)
        prod = lgb.train(PARAMS, lgb.Dataset(X_all.loc[allm, cols], y[allm]), n_iter)
        prod.save_model(str(MOD / f"{key}.txt"))
        meta["models"][key] = dict(feature_set=set_name, columns=cols, n_iter=n_iter, test_metrics=results[set_name])
    with open(MOD / "scenario_model.pkl", "wb") as fh:
        pickle.dump(sm, fh)
    (MOD / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "scenarios"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
