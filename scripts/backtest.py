"""Арилжааны туршилт: загвар шүүлтүүр болж ашиг нэмэх үү?

3 анхдагч стратеги × 3 шүүлтүүр, test үе (2023-01 .. 2025-04):
  стратеги : breakout (тренд), meanrev (эргэлт), ict_sweep (эргэлт)
  шүүлтүүр : none        — дохио бүрийг авна
             scenario    — хувилбарын загвар (train-д л сургасан) тухайн стратегид тохирох
                           хувилбаруудын магадлалыг ердийнхөөс ӨНДӨР гэж үзвэл л авна:
                             тренд стратеги  -> P(тогтвортой өсөлт)+P(тогтвортой уналт)
                             эргэлтийн       -> P(range)+P(унаад өснө)+P(өсөөд унана)
                           Босго = сургалтын давтамж (тааруулаагүй).
             meta        — López de Prado meta-labeling: train үеийн арилжаанууд дээр
                           "энэ дохио ашигтай байх уу" гэсэн LightGBM, valid-аар early stopping.
Зардал: 1 пип/арилжаа + мэдээний цагт +1.5 пип. Мэдрэмж: 0.5 / 1 / 2 пип.
Олон туршилтын засвар: 9 хувилбараас хамгийн сайныг нь Deflated Sharpe Ratio-оор шалгана.

Ажиллуулах: python scripts/backtest.py
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

from fxmodel import backtest as bt  # noqa: E402
from fxmodel.features import NEWS_SCHEDULE_COLS, build_features, group_columns  # noqa: E402
from fxmodel.scenarios import ScenarioModel, future_paths  # noqa: E402

H = 24
TRAIN_END = pd.Timestamp("2022-01-01", tz="UTC")
VALID_END = pd.Timestamp("2023-01-01", tz="UTC")
REP = ROOT / "reports"
PARAMS = dict(objective="multiclass", learning_rate=0.03, num_leaves=31, min_data_in_leaf=300,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, seed=0, num_threads=2)
META_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                   verbose=-1, seed=0, num_threads=2)
KIND = {"breakout": "trend", "meanrev": "reversal", "ict_sweep": "reversal"}


def main():
    REP.mkdir(exist_ok=True)
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    X = build_features(df)
    groups = group_columns(df)
    idx = df.index
    end = idx.max()
    news_hours = set(idx[df["ev_count"] > 0])

    # ---- Хувилбарын загвар: train-д л сургана (test-д out-of-sample магадлал) ----
    paths = future_paths(df, H)
    ok = paths.notna().all(axis=1) & X[groups["A"]].notna().all(axis=1) & (idx >= idx[300])
    purge = pd.Timedelta(hours=H)
    tr = ok & (idx < TRAIN_END - purge)
    va = ok & (idx >= TRAIN_END) & (idx < VALID_END - purge)
    sm = ScenarioModel().fit(paths[tr])
    y = pd.Series(-1, index=idx)
    y[ok] = sm.assign(paths[ok])
    cols = groups["A"] + NEWS_SCHEDULE_COLS
    m = lgb.train(PARAMS | {"num_class": sm.k}, lgb.Dataset(X.loc[tr, cols], y[tr]), 3000,
                  valid_sets=[lgb.Dataset(X.loc[va, cols], y[va])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    live = idx >= TRAIN_END                         # valid + test (out-of-sample)
    P = pd.DataFrame(m.predict(X.loc[live, cols], num_iteration=m.best_iteration), index=idx[live])
    p_trend = P[1] + P[2]
    p_rev = P[0] + P[3] + P[4]
    b = sm.base_rates
    scen_take = {"trend": p_trend > b[1] + b[2], "reversal": p_rev > b[0] + b[3] + b[4]}

    signals = {"breakout": bt.signal_breakout(df), "meanrev": bt.signal_meanrev(df), "ict_sweep": bt.signal_ict_sweep(X)}
    meta_cols = groups["A"] + groups["C"] + NEWS_SCHEDULE_COLS

    results, all_trades, trial_srs = {}, [], {}
    cost_grid = (0.5, 1.0, 2.0)
    for name, sig in signals.items():
        # ---- Meta-label загвар: train/valid үеийн арилжаан дээр сургана ----
        base_tr = bt.simulate(df[idx < VALID_END], sig[idx < VALID_END], news_hours=news_hours)
        base_tr = base_tr[base_tr["exit_ts"] < VALID_END - purge]
        feats = X.loc[base_tr["signal_ts"], meta_cols].assign(side=base_tr["side"].to_numpy())
        lab = (base_tr["pnl_pips"] > 0).astype(int).to_numpy()
        in_tr = (base_tr["signal_ts"] < TRAIN_END - purge).to_numpy()
        in_va = (base_tr["signal_ts"] >= TRAIN_END).to_numpy()
        mm = lgb.train(META_PARAMS, lgb.Dataset(feats[in_tr], lab[in_tr]), 2000,
                       valid_sets=[lgb.Dataset(feats[in_va], lab[in_va])],
                       callbacks=[lgb.early_stopping(100, verbose=False)])
        thr = float(lab[in_tr].mean())                # сургалтын ялалтын хувь — тааруулаагүй босго
        sig_idx = idx[(sig != 0) & (idx >= VALID_END)]
        mf = X.loc[sig_idx, meta_cols].assign(side=sig[sig_idx].to_numpy())
        meta_take = pd.Series(mm.predict(mf, num_iteration=mm.best_iteration) > thr, index=sig_idx)

        takes = {"none": None, "scenario": scen_take[KIND[name]], "meta": meta_take}
        for fname, take in takes.items():
            key = f"{name}|{fname}"
            results[key] = {"strategy": name, "filter": fname, "kind": KIND[name]}
            for cost in cost_grid:
                cfg = bt.BarrierConfig(cost_pips=cost)
                t = bt.simulate(df[idx >= VALID_END], sig[idx >= VALID_END], take, cfg, news_hours)
                s = bt.summarize(t, VALID_END, end)
                results[key][f"cost_{cost}"] = s
                if cost == 1.0:
                    trial_srs[key] = s["sharpe_daily"]
                    all_trades.append(t.assign(strategy=name, filter=fname))
        # Мэдээлэл: шүүлтүүргүй стратеги бүх түүх дээр (2015–2022, загваргүй тул in-sample биш)
        hist = bt.simulate(df[idx < VALID_END], sig[idx < VALID_END], None, bt.BarrierConfig(), news_hours)
        results[f"{name}|none"]["history_2015_2022"] = bt.summarize(hist, idx.min(), VALID_END)

    # ---- Олон туршилтын засвар (Deflated Sharpe) ----
    best = max(trial_srs, key=trial_srs.get)
    s = results[best]["cost_1.0"]
    dsr = bt.deflated_sharpe(s["sharpe_daily"], s["n_days"], s["skew"], s["kurt"], list(trial_srs.values()))
    summary = dict(best=best, best_sharpe_annual=s["sharpe_annual"], best_psr=s["psr"], best_dsr=dsr,
                   n_trials=len(trial_srs), test=[str(VALID_END), str(end)],
                   scenario_model_best_iter=int(m.best_iteration))
    out = dict(summary=summary, results=results)
    (REP / "backtest_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    trades = pd.concat(all_trades, ignore_index=True)
    trades.to_csv(REP / "backtest_trades.csv", index=False)

    # ---- Хүснэгт ----
    rows = []
    for key, r in results.items():
        c = r["cost_1.0"]
        rows.append(dict(config=key, trades=c["trades"], win=round(100 * c["win_rate"], 1) if c["trades"] else None,
                         avg_pips=round(c["avg_pips"], 2) if c["trades"] else None, total=round(c["total_pips"]),
                         sharpe=round(c["sharpe_annual"], 2), maxdd=round(c["max_dd_pips"]),
                         psr=round(c["psr"], 3), total_0_5=round(r["cost_0.5"]["total_pips"]),
                         total_2_0=round(r["cost_2.0"]["total_pips"])))
    table = pd.DataFrame(rows)
    table.to_csv(REP / "backtest_table.csv", index=False)
    print(table.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))

    # ---- Equity муруй ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    colors = {"none": "#888", "scenario": "#3b6fb6", "meta": "#e3a33b"}
    for ax, name in zip(axes, signals):
        for fname in ("none", "scenario", "meta"):
            t = trades[(trades.strategy == name) & (trades["filter"] == fname)]
            d = bt.daily_pnl(t, VALID_END, end).cumsum()
            ax.plot(d.index, d.values, color=colors[fname],
                    label=f"{fname} ({len(t)} арилжаа, {d.iloc[-1]:+.0f} пип)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30)
    axes[0].set_ylabel("Хуримтлагдсан ашиг, пип (зардал 1 пип)")
    fig.suptitle("Backtest 2023-01 – 2025-04 (out-of-sample), шүүлтүүргүй vs хувилбарын загвар vs meta-label")
    fig.tight_layout()
    fig.savefig(REP / "backtest_equity.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
