"""Ирэх 24 цагийн хүрээний (пип) загвар: HAR baseline vs LightGBM quantile.

Хуваалт train.py-тэй ижил (train 2015–2021, valid 2022, test 2023–2025-04, 24ц purge).
Загварууд (log хүрээг таамаглана):
  persistence : өнгөрсөн 24 цагийн хүрээ
  HAR         : OLS(log хүрээ ~ өдөр + 7 хоног + сар + цагийн улирал)
  LGBM        : HAR + суурь шинж (A)                   — мэдээгүй
  LGBM+S      : HAR + A + өндөр нөлөөтэй мэдээний хуваарь
  (+X         : бусад зах зээл, data/raw/markets байвал)
LightGBM нь 10/50/90 хувийн quantile-ийг гаргана. 80% интервалын хамрах хувь (coverage)-ийг шалгана.

Ажиллуулах: python scripts/train_vol.py
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

from fxmodel.cross_asset import cross_features, load_markets  # noqa: E402
from fxmodel.features import NEWS_SCHEDULE_COLS, build_features, group_columns  # noqa: E402
from fxmodel.volatility import future_range, har_features, pinball, past_range  # noqa: E402

H = 24
from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
REP, MOD = ROOT / "reports", ROOT / "models"
QS = (0.1, 0.5, 0.9)
PARAMS = dict(objective="quantile", learning_rate=0.03, num_leaves=31, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=0, num_threads=2)
HAR_COLS = ["har_d", "har_w", "har_m", "har_hour_sin", "har_hour_cos"]


def ols(Xtr, ytr):
    A = np.c_[np.ones(len(Xtr)), Xtr]
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return lambda X: np.c_[np.ones(len(X)), X] @ beta, beta


def bootstrap_mae_diff(y, a, b, weeks, n=500, seed=0):
    """MAE(a) − MAE(b)-ийн 95% интервал, долоо хоногоор блок bootstrap (>0 = b сайн)."""
    rng = np.random.default_rng(seed)
    d = pd.Series(np.abs(y - a) - np.abs(y - b)).groupby(weeks).agg(["sum", "count"])
    out = []
    for _ in range(n):
        s = d.sample(len(d), replace=True, random_state=rng.integers(1 << 31))
        out.append(s["sum"].sum() / s["count"].sum())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main():
    REP.mkdir(exist_ok=True)
    MOD.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    X = build_features(df).join(har_features(df))
    groups = group_columns(df)
    sets = {"LGBM": HAR_COLS + groups["A"], "LGBM+S": HAR_COLS + groups["A"] + NEWS_SCHEDULE_COLS}
    mk = load_markets(ROOT / "data/raw/markets")
    if mk:
        xf = cross_features(df, mk)
        X = X.join(xf)
        sets["LGBM+S+X"] = sets["LGBM+S"] + list(xf.columns)
        print("Бусад зах зээл:", sorted(mk))

    y_pips = future_range(df, H)
    y = np.log(y_pips.clip(lower=1))
    idx = df.index
    ok = y_pips.notna() & X[HAR_COLS + groups["A"]].notna().all(axis=1)
    purge = pd.Timedelta(hours=H)
    tr = ok & (idx < TRAIN_END - purge)
    va = ok & (idx >= TRAIN_END) & (idx < VALID_END - purge)
    te = ok & (idx >= VALID_END) & (idx < TEST_END - purge)
    yte = y[te].to_numpy()
    weeks = idx[te].tz_localize(None).to_period("W").astype(str)

    pred, res = {}, {}
    # Persistence
    pred["persistence"] = {0.5: np.log(past_range(df).clip(lower=1))[te].to_numpy()}
    # HAR
    f_har, beta = ols(X.loc[tr, HAR_COLS].to_numpy(), y[tr].to_numpy())
    resid = y[va].to_numpy() - f_har(X.loc[va, HAR_COLS].to_numpy())
    q_lo, q_hi = np.quantile(resid, [0.1, 0.9])          # интервалыг valid-ийн үлдэгдлээс
    har_te = f_har(X.loc[te, HAR_COLS].to_numpy())
    pred["HAR"] = {0.1: har_te + q_lo, 0.5: har_te, 0.9: har_te + q_hi}
    res["HAR_coef"] = dict(zip(["const"] + HAR_COLS, map(float, beta)))

    boosters = {}
    for name, cols in sets.items():
        pred[name], boosters[name] = {}, {}
        for a in QS:
            m = lgb.train(PARAMS | {"alpha": a}, lgb.Dataset(X.loc[tr, cols], y[tr]), 3000,
                          valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
            pred[name][a] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
            boosters[name][a] = (m, cols)

    ss_tot = np.sum((yte - yte.mean()) ** 2)
    for name, p in pred.items():
        q50 = p[0.5]
        r = dict(mae_log=float(np.mean(np.abs(yte - q50))), r2_log=float(1 - np.sum((yte - q50) ** 2) / ss_tot),
                 mae_pips=float(np.mean(np.abs(np.exp(yte) - np.exp(q50)))),
                 median_abs_pct_err=float(np.median(np.abs(np.exp(q50) / np.exp(yte) - 1)) * 100))
        if 0.1 in p:
            r["coverage80"] = float(np.mean((yte >= p[0.1]) & (yte <= p[0.9])))
            r["avg_width_pips"] = float(np.mean(np.exp(p[0.9]) - np.exp(p[0.1])))
            r["pinball_avg"] = float(np.mean([pinball(yte, p[a], a) for a in QS]))
        if name != "HAR":
            r["mae_vs_HAR_ci95"] = bootstrap_mae_diff(yte, pred["HAR"][0.5], q50, weeks)
        res[name] = r
    if "LGBM+S" in pred:
        res["LGBM+S"]["mae_vs_LGBM_ci95"] = bootstrap_mae_diff(yte, pred["LGBM"][0.5], pred["LGBM+S"][0.5], weeks)

    # Мэдээтэй/мэдээгүй цагуудаар
    soon = (X.loc[te, "hours_to_next_event"] <= 24).to_numpy()
    for lbl, msk in (("news_within_24h", soon), ("no_news_24h", ~soon)):
        res[f"by_{lbl}"] = {n: float(np.mean(np.abs(yte[msk] - pred[n][0.5][msk]))) for n in ("HAR", "LGBM", "LGBM+S")}
    res["test_median_range_pips"] = float(np.median(np.exp(yte)))
    res["split"] = dict(train=int(tr.sum()), valid=int(va.sum()), test=int(te.sum()))
    (REP / "vol_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "HAR_coef"}, ensure_ascii=False, indent=1))

    # ---- График: test-ийн хэсэг (2024-03), бодит vs таамаг ----
    best = "LGBM+S"
    s = pd.DataFrame({"real": np.exp(yte), "q10": np.exp(pred[best][0.1]), "q50": np.exp(pred[best][0.5]),
                      "q90": np.exp(pred[best][0.9]), "har": np.exp(pred["HAR"][0.5])}, index=idx[te])
    w = s["2024-03-01":"2024-04-15"].iloc[::4]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(w.index, w.q10, w.q90, alpha=0.25, label="LGBM+S 80% интервал")
    ax.plot(w.index, w.q50, lw=1.6, label="LGBM+S медиан")
    ax.plot(w.index, w.har, lw=1, ls="--", color="grey", label="HAR")
    ax.plot(w.index, w.real, lw=1, color="black", label="Бодит 24ц хүрээ")
    ax.set_ylabel("Ирэх 24 цагийн хүрээ, пип")
    ax.set_title("Хүрээний таамаглал (test, 2024-03 – 2024-04)")
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(REP / "vol_forecast.png", dpi=140)
    plt.close(fig)

    # ---- Бодит цагийн загварууд: train+valid+test бүгдээр дахин сургана ----
    allm = ok & (idx < idx[te].max() - purge)
    meta = {"horizon": H, "quantiles": list(QS), "models": {}}
    for key, name in (("vol_news", "LGBM+S"), ("vol_nonews", "LGBM")):
        meta["models"][key] = {}
        for a in QS:
            m, cols = boosters[name][a]
            n_iter = max(int(m.best_iteration * 1.15), 20)
            prod = lgb.train(PARAMS | {"alpha": a}, lgb.Dataset(X.loc[allm, cols], y[allm]), n_iter)
            fn = f"{key}_q{int(a * 100)}.txt"
            prod.save_model(str(MOD / fn))
            meta["models"][key][str(a)] = dict(file=fn, columns=cols)
    (MOD / "vol_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
