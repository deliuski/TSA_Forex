"""Арилжааны туршилт (backtest): анхдагч стратеги + шүүлтүүр, бодит зардал, Deflated Sharpe Ratio.

Зарчим (López de Prado, AFML 3-р бүлэг — meta-labeling):
  Загвар маань чиглэлийг таамаглаж чаддаггүй (AUC≈0.50) тул өөрөө дохио өгөхгүй.
  Харин энгийн, ил тод АНХДАГЧ стратеги дохио өгнө. Загвар нь тухайн дохиог авах уу, алгасах уу
  гэдгийг шийдэх ШҮҮЛТҮҮР болно.

Бодит байдлын дүрмүүд:
  * Дохио t лааны ХААЛТАД гарна, орох үнэ = t+1 лааны НЭЭЛТ (spread-ийн тал нь нэмэгдэнэ).
  * Гарах: triple barrier — ашгийн (TP), алдагдлын (SL) хязгаар ATR[t]-ээр, 24 цагийн хугацааны хязгаар.
  * Нэг лаан дотор TP ба SL хоёулаа хүрсэн бол SL гэж үзнэ (болгоомжтой таамаглал).
  * Нэг стратеги нэг удаад ганц позицтой (давхардсан дохиог алгасна).
  * Зардал: spread+шимтгэл пипээр. Өндөр нөлөөтэй мэдээний цагт нэмэлт зардал.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import atr

PIP = 1e-4


# ---------------- Анхдагч стратегиуд (параметр цөөн, урьдчилан тогтоосон) ----------------
def signal_breakout(df: pd.DataFrame, n: int = 24) -> pd.Series:
    """Donchian breakout: хаалт сүүлийн n цагийн дээд/доод түвшнийг давбал тэр зүг рүү (трендийг дагах)."""
    hi = df["high"].rolling(n).max().shift(1)
    lo = df["low"].rolling(n).min().shift(1)
    s = pd.Series(0, index=df.index)
    s[df["close"] > hi] = 1
    s[df["close"] < lo] = -1
    return s


def signal_meanrev(df: pd.DataFrame, n: int = 20, z: float = 2.0) -> pd.Series:
    """Bollinger mean reversion: хэт өндөрт зарна, хэт доор авна (эргэлтэд бооцоо тавих)."""
    c = df["close"]
    zz = (c - c.rolling(n).mean()) / c.rolling(n).std()
    s = pd.Series(0, index=df.index)
    s[zz > z] = -1
    s[zz < -z] = 1
    return s


def signal_ict_sweep(feats: pd.DataFrame) -> pd.Series:
    """ICT liquidity sweep: swing low-ийн доор орж буцаж хаасан бол авна, swing high бол зарна."""
    s = pd.Series(0, index=feats.index)
    lo = feats["sweep_low_recent"].diff().fillna(0) > 0      # шинээр гарсан sweep
    hi = feats["sweep_high_recent"].diff().fillna(0) > 0
    s[lo & ~hi] = 1
    s[hi & ~lo] = -1
    return s


# ---------------- Triple barrier симуляц ----------------
@dataclass
class BarrierConfig:
    tp_atr: float = 1.5
    sl_atr: float = 1.5
    max_hours: int = 24
    cost_pips: float = 1.0          # нэг арилжааны нийт зардал (spread + шимтгэл)
    news_extra_pips: float = 1.5    # орох/гарах цагт өндөр нөлөөтэй мэдээ байвал нэмэлт


def simulate(df: pd.DataFrame, signal: pd.Series, take: pd.Series | None = None,
             cfg: BarrierConfig | None = None, news_hours: set | None = None) -> pd.DataFrame:
    """Дохио бүрийг triple barrier-ээр гүйцэтгэнэ. take=False бол тухайн дохиог алгасна."""
    cfg = cfg or BarrierConfig()
    a = atr(df).to_numpy()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = df.index
    sig = signal.reindex(idx).fillna(0).to_numpy()
    ok = np.ones(len(idx), bool) if take is None else take.reindex(idx).fillna(False).to_numpy(bool)
    news_hours = news_hours or set()
    trades, i, n = [], 0, len(idx)
    while i < n - 2:
        if sig[i] == 0 or not ok[i] or np.isnan(a[i]):
            i += 1
            continue
        side, e = int(sig[i]), i + 1
        # Амралтын өдрийн өмнө (лааны дараалал тасарсан) орохгүй
        if (idx[e] - idx[i]) > pd.Timedelta(hours=2):
            i += 1
            continue
        entry = o[e]
        tp = entry + side * cfg.tp_atr * a[i]
        sl = entry - side * cfg.sl_atr * a[i]
        exit_px, j, reason = None, e, "time"
        last = min(n - 1, e + cfg.max_hours - 1)
        for j in range(e, last + 1):
            if (idx[j] - idx[e]) >= pd.Timedelta(hours=cfg.max_hours):
                j -= 1
                break
            hit_sl = (l[j] <= sl) if side > 0 else (h[j] >= sl)
            hit_tp = (h[j] >= tp) if side > 0 else (l[j] <= tp)
            if hit_sl:
                exit_px, reason = sl, "sl"
                break
            if hit_tp:
                exit_px, reason = tp, "tp"
                break
        if exit_px is None:
            exit_px = c[j]
        cost = cfg.cost_pips
        if idx[i] in news_hours or idx[e] in news_hours or idx[j] in news_hours:
            cost += cfg.news_extra_pips
        pnl = side * (exit_px - entry) / PIP - cost
        trades.append(dict(signal_ts=idx[i], entry_ts=idx[e], exit_ts=idx[j], side=side, entry=entry,
                           exit=exit_px, reason=reason, pnl_pips=pnl, atr_pips=a[i] / PIP))
        i = j + 1
    return pd.DataFrame(trades)


# ---------------- Үзүүлэлтүүд ----------------
def daily_pnl(trades: pd.DataFrame, start, end) -> pd.Series:
    days = pd.date_range(pd.Timestamp(start).floor("D"), pd.Timestamp(end).floor("D"), freq="D", tz="UTC")
    days = days[days.dayofweek < 5]
    if trades.empty:
        return pd.Series(0.0, index=days)
    s = trades.groupby(trades["exit_ts"].dt.floor("D"))["pnl_pips"].sum()
    # Ням гарагийн оройн нээлтийн лаанд хаагдсан арилжааг алдахгүй
    return s.reindex(days.union(s.index), fill_value=0.0)


def sharpe(x: pd.Series) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float, sr_star: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012): жинхэнэ SR > sr_star байх магадлал. sr = нэг үеийн (өдрийн)."""
    den = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_star) * np.sqrt(n - 1) / den))


def deflated_sharpe(sr: float, n: int, skew: float, kurt: float, trial_srs: list[float]) -> float:
    """DSR (Bailey & López de Prado 2014): N туршилтаас хамгийн сайныг сонгосны хазайлтыг засна."""
    N = len(trial_srs)
    v = np.var(trial_srs, ddof=1) if N > 1 else 0.0
    g = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / N) + g * norm.ppf(1 - 1 / (N * np.e))) if N > 1 else 0.0
    return probabilistic_sharpe(sr, n, skew, kurt, sr0)


def summarize(trades: pd.DataFrame, start, end) -> dict:
    d = daily_pnl(trades, start, end)
    eq = d.cumsum()
    out = dict(trades=int(len(trades)),
               win_rate=float((trades["pnl_pips"] > 0).mean()) if len(trades) else np.nan,
               avg_pips=float(trades["pnl_pips"].mean()) if len(trades) else np.nan,
               total_pips=float(trades["pnl_pips"].sum()) if len(trades) else 0.0,
               sharpe_daily=sharpe(d), sharpe_annual=sharpe(d) * np.sqrt(252),
               max_dd_pips=float((eq - eq.cummax()).min()), n_days=int(len(d)),
               skew=float(d.skew()), kurt=float(d.kurt() + 3))
    out["psr"] = probabilistic_sharpe(out["sharpe_daily"], len(d), out["skew"], out["kurt"])
    if len(trades):
        t = trades["pnl_pips"]
        out["t_stat"] = float(t.mean() / (t.std(ddof=1) / np.sqrt(len(t)))) if len(t) > 1 else np.nan
    return out
