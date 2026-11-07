"""Туршилт: medium нөлөөтэй мэдээний хуваарийг нэмэхэд загвар сайжрах уу?

Харьцуулалт (train.py-тэй ижил хуваалт, параметр):
  A+S        : суурь + өндөр нөлөөтэй мэдээний хуваарь (одоогийн бодит цагийн загвар)
  A+S+M      : + medium мэдээний хуваарь (дараагийн medium мэдээ хүртэлх хугацаа, тоо)
  A+S(h+m)   : өндөр ба medium-ийг нэг "мэдээ" гэж нийлүүлсэн хуваарь
Ажиллуулах: python scripts/exp_medium_news.py
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

from fxdata import calendar, merge  # noqa: E402
from fxmodel.features import NEWS_SCHEDULE_COLS, build_features, group_columns  # noqa: E402
from fxmodel.scenarios import ScenarioModel, future_paths  # noqa: E402
from train import PARAMS, TRAIN_END, VALID_END, H, block_bootstrap_diff  # noqa: E402


def sched(feats: pd.DataFrame, suffix: str) -> pd.DataFrame:
    f = feats[["ev_count", "hours_since_event", "hours_to_next_event"]].copy()
    f["hours_since_event"] = f["hours_since_event"].clip(upper=168)
    f["hours_to_next_event"] = f["hours_to_next_event"].clip(upper=168)
    f["event_within_24h"] = (f["hours_to_next_event"] <= 24).astype(int)
    return f.add_suffix(suffix)


def main():
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    ev, _ = calendar.load_calendar(ROOT / "data/raw/calendar/forex_factory.csv", "Asia/Tehran")
    ev = ev[ev["ts_utc"] < df.index.max() + pd.Timedelta(hours=1)]
    candles = df[["open", "high", "low", "close", "volume"]]
    med = sched(merge.news_features(candles, ev, impact="medium"), "_med")
    hm = ev[ev["impact"].isin(["high", "medium"])].assign(impact="hm")
    both = sched(merge.news_features(candles, hm, impact="hm"), "_hm")

    X = build_features(df).join(med).join(both)
    groups = group_columns(df)
    paths = future_paths(df, H)
    idx = df.index
    ok = paths.notna().all(axis=1) & X[groups["A"]].notna().all(axis=1) & (idx >= idx[300])
    purge = pd.Timedelta(hours=H)
    tr, va, te = ok & (idx < TRAIN_END - purge), ok & (idx >= TRAIN_END) & (idx < VALID_END - purge), ok & (idx >= VALID_END)
    sm = ScenarioModel().fit(paths[tr])
    y = pd.Series(-1, index=idx)
    y[ok] = sm.assign(paths[ok])
    yte = y[te].to_numpy()

    sets = {
        "A+S": groups["A"] + NEWS_SCHEDULE_COLS,
        "A+S+M": groups["A"] + NEWS_SCHEDULE_COLS + list(med.columns),
        "A+S(h+m)": groups["A"] + list(both.columns),
    }
    probs, res = {}, {}
    for name, cols in sets.items():
        m = lgb.train(PARAMS | {"num_class": sm.k}, lgb.Dataset(X.loc[tr, cols], y[tr]), 3000,
                      valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
        probs[name] = m.predict(X.loc[te, cols], num_iteration=m.best_iteration)
        res[name] = dict(logloss=log_loss(yte, probs[name], labels=range(sm.k)), best_iter=int(m.best_iteration))
    base = log_loss(yte, np.tile(sm.base_rates, (te.sum(), 1)), labels=range(sm.k))
    weeks = idx[te].tz_localize(None).to_period("W").astype(str)
    for name in sets:
        res[name]["gain_vs_baseline_pct"] = 100 * (1 - res[name]["logloss"] / base)
        if name != "A+S":
            lo, hi = block_bootstrap_diff(yte, probs["A+S"], probs[name], weeks)
            res[name]["vs_A+S_pct_ci95"] = [100 * lo / base, 100 * hi / base]
    out = ROOT / "reports" / "exp_medium_news.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
