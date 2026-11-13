"""Туршилт: лааны бүтэц (FLF-LSTM-ийн санаа) нэмэр болох уу?

(А) Лааны хэлбэрийн шинжүүд (K): дээд/доод сүүдэр, бие, хаалтын байрлал, inside bar, engulfing,
    сүүлийн 3 лааны чиглэлийн дараалал. Хувилбарын загвар (A+S vs A+S+K) болон хүрээний
    загвар (LGBM+S vs LGBM+S+K) дээр ablation хийнэ.
(Б) FLF-LSTM-ийн зорилтыг ЗӨВ аргачлалаар: дараагийн H4 лааны
      дээд сүүдэр  up = High[t+1] − Close[t],  доод  dn = Close[t] − Low[t+1],  хаалт  Close[t+1] − Close[t]
    (пип). Baseline: NAIVE-1 (дараагийн лаа = одоогийн лаа), NAIVE-2 (Open=Close, сүүдэр = train-ийн дундаж).
    Загвар: LightGBM (L1 алдаа), хугацааны дарааллаар хуваасан (train 2015–2021, valid 2022, test 2023–2025-04).

Ажиллуулах: python scripts/exp_candle.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from fxmodel.features import NEWS_SCHEDULE_COLS, atr, base_features, build_features, group_columns  # noqa: E402
from fxmodel.scenarios import ScenarioModel, future_paths  # noqa: E402
from fxmodel.volatility import future_range, har_features  # noqa: E402
from train import PARAMS, TRAIN_END, VALID_END, block_bootstrap_diff  # noqa: E402

PIP = 1e-4


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    a = atr(df)
    rng = (h - l).replace(0, np.nan)
    f = pd.DataFrame(index=df.index)
    for k in (0, 1, 2):
        oo, hh, ll, cc, rr = o.shift(k), h.shift(k), l.shift(k), c.shift(k), rng.shift(k)
        f[f"k{k}_upwick_atr"] = (hh - np.maximum(oo, cc)) / a
        f[f"k{k}_dnwick_atr"] = (np.minimum(oo, cc) - ll) / a
        f[f"k{k}_body_atr"] = (cc - oo) / a
        f[f"k{k}_body_ratio"] = (cc - oo).abs() / rr
        f[f"k{k}_close_pos"] = (cc - ll) / rr
    f["inside_bar"] = ((h < h.shift(1)) & (l > l.shift(1))).astype(int)
    bull_eng = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear_eng = (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))
    f["engulfing"] = bull_eng.astype(int) - bear_eng.astype(int)
    d = np.sign(c - o)
    f["dir_seq3"] = (d + 1) * 9 + (d.shift(1) + 1) * 3 + (d.shift(2) + 1)      # 27 төрлийн дараалал
    return f


def part_a(df):
    X = build_features(df).join(candle_features(df)).join(har_features(df))
    g = group_columns(df)
    K = list(candle_features(df.iloc[:50]).columns)
    idx = df.index
    purge = pd.Timedelta(hours=24)
    out = {}
    # Хувилбарын загвар
    paths = future_paths(df, 24)
    ok = paths.notna().all(axis=1) & X[g["A"]].notna().all(axis=1) & (idx >= idx[300])
    tr, va, te = ok & (idx < TRAIN_END - purge), ok & (idx >= TRAIN_END) & (idx < VALID_END - purge), ok & (idx >= VALID_END)
    sm = ScenarioModel().fit(paths[tr])
    y = pd.Series(-1, index=idx)
    y[ok] = sm.assign(paths[ok])
    yte = y[te].to_numpy()
    weeks = idx[te].tz_localize(None).to_period("W").astype(str)
    base = log_loss(yte, np.tile(sm.base_rates, (te.sum(), 1)), labels=range(sm.k))
    probs = {}
    for name, cols in {"A+S": g["A"] + NEWS_SCHEDULE_COLS, "A+S+K": g["A"] + NEWS_SCHEDULE_COLS + K}.items():
        m = lgb.train(PARAMS | {"num_class": sm.k}, lgb.Dataset(X.loc[tr, cols], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])], callbacks=[lgb.early_stopping(150, verbose=False)])
        probs[name] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
        out[f"scenario_{name}_gain_pct"] = 100 * (1 - log_loss(yte, probs[name], labels=range(sm.k)) / base)
    lo, hi = block_bootstrap_diff(yte, probs["A+S"], probs["A+S+K"], weeks)
    out["scenario_K_vs_A+S_pct_ci95"] = [100 * lo / base, 100 * hi / base]
    # Хүрээний загвар
    yr = np.log(future_range(df, 24).clip(lower=1))
    har = ["har_d", "har_w", "har_m", "har_hour_sin", "har_hour_cos"]
    okr = yr.notna() & X[har + g["A"]].notna().all(axis=1)
    tr, va, te = okr & (idx < TRAIN_END - purge), okr & (idx >= TRAIN_END) & (idx < VALID_END - purge), okr & (idx >= VALID_END)
    qp = dict(PARAMS, objective="quantile", alpha=0.5)
    qp.pop("num_class", None)
    pr = {}
    for name, cols in {"LGBM+S": har + g["A"] + NEWS_SCHEDULE_COLS, "LGBM+S+K": har + g["A"] + NEWS_SCHEDULE_COLS + K}.items():
        m = lgb.train(qp, lgb.Dataset(X.loc[tr, cols], yr[tr]), 3000, valid_sets=[lgb.Dataset(X.loc[va, cols], yr[va])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pr[name] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
        out[f"range_{name}_mae_log"] = float(np.mean(np.abs(yr[te].to_numpy() - pr[name])))
    out["range_K_improvement_mae_log"] = out["range_LGBM+S_mae_log"] - out["range_LGBM+S+K_mae_log"]
    return out


def part_b(df):
    """FLF-ийн зорилт зөв аргачлалаар: дараагийн H4 лаа."""
    h4 = df[["open", "high", "low", "close", "volume"]].resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    c = h4["close"]
    tgt = pd.DataFrame({"up": (h4["high"].shift(-1) - c) / PIP, "dn": (c - h4["low"].shift(-1)) / PIP,
                        "dc": (c.shift(-1) - c) / PIP, "do": (h4["open"].shift(-1) - c) / PIP})
    nxt = pd.Series(h4.index).shift(-1).to_numpy() - h4.index.to_numpy()
    tgt[nxt != np.timedelta64(4, "h")] = np.nan                       # дараагийн лаа шууд залгах ёстой
    X = base_features(h4).join(candle_features(h4)).join(har_features(h4))
    idx = h4.index
    ok = tgt.notna().all(axis=1) & X.notna().all(axis=1)
    tr, va, te = ok & (idx < TRAIN_END), ok & (idx >= TRAIN_END) & (idx < VALID_END), ok & (idx >= VALID_END)
    out = {"n_test_h4": int(te.sum())}
    # Үнийн түвшний MAE (пип) — FLF-ийн хүснэгттэй ижил хэлбэр: Open, High, Low, Close
    cur = h4.loc[te]
    true = pd.DataFrame({"Open": h4["open"].shift(-1), "High": h4["high"].shift(-1),
                         "Low": h4["low"].shift(-1), "Close": h4["close"].shift(-1)}).loc[te]
    def mae(pred):  # noqa: E306
        return {k: float((pred[k] - true[k]).abs().mean() / PIP) for k in true}
    res = {"NAIVE-1 (дараагийн = одоогийн OHLC)": mae(pd.DataFrame({"Open": cur.open, "High": cur.high, "Low": cur.low, "Close": cur.close}))}
    up_m, dn_m = tgt.loc[tr, "up"].mean(), tgt.loc[tr, "dn"].mean()
    res["NAIVE-2 (O=C, C=C, сүүдэр=дундаж)"] = mae(pd.DataFrame({"Open": cur.close, "High": cur.close + up_m * PIP,
                                                                  "Low": cur.close - dn_m * PIP, "Close": cur.close}))
    pred = {}
    for k in ("up", "dn", "dc"):
        m = lgb.train(dict(objective="l1", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0, num_threads=2),
                      lgb.Dataset(X[tr], tgt.loc[tr, k]), 3000, valid_sets=[lgb.Dataset(X[va], tgt.loc[va, k])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        pred[k] = m.predict(X[te], num_iteration=m.best_iteration)
    res["LightGBM (сүүдэр, хаалтын өөрчлөлт)"] = mae(pd.DataFrame({"Open": cur.close, "High": cur.close + pred["up"] * PIP,
                                                                     "Low": cur.close - pred["dn"] * PIP,
                                                                     "Close": cur.close + pred["dc"] * PIP}))
    for r in res.values():
        r["avg"] = float(np.mean([r[k] for k in ("Open", "High", "Low", "Close")]))
    out["h4_next_candle_mae_pips"] = res
    # Сүүдрийн хэмжээ таамаглагддаг уу (корреляц), хаалтын чиглэл таамаглагддаг уу
    out["corr_pred_true_up"] = float(np.corrcoef(pred["up"], tgt.loc[te, "up"])[0, 1])
    out["corr_pred_true_dn"] = float(np.corrcoef(pred["dn"], tgt.loc[te, "dn"])[0, 1])
    out["direction_hit_close"] = float(np.mean(np.sign(pred["dc"]) == np.sign(tgt.loc[te, "dc"])))
    return out


def main():
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    res = {"A_candle_features": part_a(df), "B_flf_task_done_right": part_b(df)}
    (ROOT / "reports" / "exp_candle.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
