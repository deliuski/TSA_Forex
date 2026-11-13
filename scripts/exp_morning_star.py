"""Туршилт: H1 200 MA + Morning/Evening Star (унах лаа, жижиг лаа, хүчтэй өсөх лаа) + 2R зорилт, мөн
"1000 FTMO симуляц".

Дүрэм (урт позиц; богино нь толин тусгал):
  1. Тренд: хаалт > SMA200 (H1).
  2. Загвар: t-2 унах лаа (бие ≥ 0.5·ATR), t-1 жижиг лаа (бие ≤ 0.3·ATR),
     t хүчтэй өсөх лаа (бие ≥ 0.5·ATR, хаалт t-2-ийн биеийн дундаас дээш).
  3. Батлалт ("дараагийн лаанаас хамаарна"): t+1 лаа t-ийн дээд цэгийг эвдвэл тэр түвшинд орно (buy-stop).
     Эвдэхгүй бол захиалгыг цуцална. Харьцуулах хувилбар: t+1-ийн нээлтэд шууд орох.
  4. SL = 3 лааны доод цэг, TP = орох үнэ + 2·(орох үнэ − SL). Хугацааны хязгаар 48 цаг.
     Нэг лаанд SL ба TP хоёулаа хүрвэл SL гэж тооцно. Зардал: 1 пип + мэдээний цагт 1.5 пип.
Null: ижил SL зай, 2R бүтэцтэй, санамсаргүй цаг, санамсаргүй чиглэлд орсон арилжаанууд.
FTMO (2-Step, 1-р шат): эрсдэл 1%/арилжаа, зорилт +10%, нийт алдагдлын хязгаар −10%, өдрийн −5%.
  Арилжааны үр дүнгээс (R) санамсаргүй дахин сонголт хийж 1000 удаа давтана.

Ажиллуулах: python scripts/exp_morning_star.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fxmodel.features import atr  # noqa: E402

PIP = 1e-4
COST, NEWS_COST, MAX_H = 1.0, 1.5, 48


def find_signals(df):
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    a = atr(df).to_numpy()
    ma = df["close"].rolling(200).mean().to_numpy()
    body = c - o
    sig = []
    for t in range(202, len(df) - 2):
        if np.isnan(a[t]) or np.isnan(ma[t]):
            continue
        b2, b1, b0 = body[t - 2], body[t - 1], body[t]
        small = abs(b1) <= 0.3 * a[t]
        if c[t] > ma[t] and b2 <= -0.5 * a[t] and small and b0 >= 0.5 * a[t] and c[t] > (o[t - 2] + c[t - 2]) / 2:
            sig.append((t, 1, h[t], min(l[t - 2:t + 1])))
        if c[t] < ma[t] and b2 >= 0.5 * a[t] and small and b0 <= -0.5 * a[t] and c[t] < (o[t - 2] + c[t - 2]) / 2:
            sig.append((t, -1, l[t], max(h[t - 2:t + 1])))
    return sig


def run_trade(df, e_idx, side, entry, sl, news, rr=2.0):
    """Орох лаа e_idx (тухайн лаанд entry үнэд орсон), SL/TP хүртэл эсвэл MAX_H цаг."""
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    idx = df.index
    risk = abs(entry - sl)
    tp = entry + side * rr * risk
    j = e_idx
    out = None
    for j in range(e_idx, min(len(df), e_idx + MAX_H)):
        if (idx[j] - idx[e_idx]) >= pd.Timedelta(hours=MAX_H):
            break
        hit_sl = (l[j] <= sl) if side > 0 else (h[j] >= sl)
        hit_tp = (h[j] >= tp) if side > 0 else (l[j] <= tp)
        if hit_sl:
            out = sl
            break
        if hit_tp and j > e_idx:            # орсон лаанд TP-г тооцохгүй (лаа доторх дарааллыг мэдэхгүй)
            out = tp
            break
    if out is None:
        out = c[j]
    cost = COST + (NEWS_COST if (idx[e_idx] in news or idx[j] in news) else 0)
    pips = side * (out - entry) / PIP - cost
    return dict(entry_ts=idx[e_idx], exit_ts=idx[j], side=side, risk_pips=risk / PIP, pips=pips, R=pips / (risk / PIP))


def backtest(df, sig, news, mode):
    o, h, l = df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    idx = df.index
    trades, busy_until = [], idx[0]
    for t, side, trig, sl in sig:
        e = t + 1
        if idx[e] <= busy_until or (idx[e] - idx[t]) > pd.Timedelta(hours=2):
            continue
        if mode == "confirm":
            hit = (h[e] >= trig) if side > 0 else (l[e] <= trig)
            if not hit:
                continue
            entry = max(trig, o[e]) if side > 0 else min(trig, o[e])
        else:
            entry = o[e]
        if (entry - sl) * side <= 2 * PIP:
            continue
        tr = run_trade(df, e, side, entry, sl, news)
        trades.append(tr)
        busy_until = tr["exit_ts"]
    return pd.DataFrame(trades)


def random_null(df, n, risk_pips, news, seed=0):
    rng = np.random.default_rng(seed)
    o = df["open"].to_numpy()
    idx = df.index
    out = []
    for _ in range(n):
        e = int(rng.integers(300, len(df) - MAX_H - 1))
        if (idx[e] - idx[e - 1]) > pd.Timedelta(hours=2):
            continue
        side = int(rng.choice([-1, 1]))
        r = float(rng.choice(risk_pips)) * PIP
        out.append(run_trade(df, e, side, o[e], o[e] - side * r, news))
    return pd.DataFrame(out)


def summary(t):
    if t.empty:
        return {"trades": 0}
    R = t["R"]
    return dict(trades=int(len(t)), win_rate=float((R > 0).mean()), avg_R=float(R.mean()), total_R=float(R.sum()),
                median_risk_pips=float(t["risk_pips"].median()), p_value_avgR=float(stats.ttest_1samp(R, 0).pvalue))


def ftmo_sim(R, n=1000, risk=0.01, target=0.10, max_loss=0.10, daily=0.05, trades_per_day=1, seed=0, max_trades=2000):
    """R дарааллаас санамсаргүй сонгож challenge-ийг давтана. Буцаана: давсан хувь, дундаж арилжааны тоо."""
    rng = np.random.default_rng(seed)
    R = np.asarray(R)
    passed, n_tr = 0, []
    for _ in range(n):
        eq, k = 1.0, 0
        while k < max_trades:
            day_start = eq
            for _ in range(trades_per_day):
                eq += risk * rng.choice(R)          # тогтмол эрсдэл (эхний балансаас)
                k += 1
                if eq - day_start <= -daily:
                    break
            if eq >= 1 + target:
                passed += 1
                break
            if eq <= 1 - max_loss or eq - day_start <= -daily:
                break
        n_tr.append(k)
    return dict(pass_rate=passed / n, median_trades=float(np.median(n_tr)))


def main():
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    news = set(df.index[df["ev_count"] > 0])
    sig = find_signals(df)
    res = {"n_patterns": len(sig)}
    for mode in ("confirm", "market"):
        t = backtest(df, sig, news, mode)
        s = summary(t)
        s["by_period"] = {f"{a}-{b}": summary(t[(t.entry_ts.dt.year >= a) & (t.entry_ts.dt.year <= b)])
                          for a, b in ((2015, 2019), (2020, 2025))}
        s["ftmo_1000"] = ftmo_sim(t["R"].to_numpy())
        res[mode] = s
        if mode == "confirm":
            null = random_null(df, 5000, t["risk_pips"].to_numpy(), news)
            res["random_2R_null"] = summary(null) | {"ftmo_1000": ftmo_sim(null["R"].to_numpy())}
            t.to_csv(ROOT / "reports" / "morning_star_trades.csv", index=False)
    # Зардалгүй, ялалт 50%, 1.5R "онолын" стратеги (форумын жишээтэй харьцуулах)
    res["theory_50pct_1.5R"] = ftmo_sim(np.r_[np.full(500, 1.5), np.full(500, -1.0)])
    (ROOT / "reports" / "exp_morning_star.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
