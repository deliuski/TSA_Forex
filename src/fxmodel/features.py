"""Шинжүүд (features). Бүгд лааны ХААЛТЫН агшинд мэдэгдэх мэдээлэл дээр суурилна (look-ahead-гүй).

Бүлгүүд (ablation study-д тус тусад нь асааж/унтраана):
  A  суурь      : өгөөж, ATR, хэлбэлзлийн горим, цаг, гараг, эзлэхүүн
  B  сонгодог TA: RSI, MACD, Bollinger %b, MA-аас хол байдал
  C  ICT/SMC    : батлагдсан swing, бүтэц (HH/HL), BOS/CHoCH, liquidity sweep, FVG,
                  premium/discount, killzone
  D  мэдээ      : мэдээний тоо, surprise_z, дараагийн мэдээ хүртэлх хугацаа

ICT-ийн look-ahead урхи: swing high нь баруун талдаа n лаа гарсны дараа л батлагдана.
Тиймээс swing-ийг i цэгт биш, i+n цэгт (батлагдсан агшинд) "мэдэгдсэн" гэж тооцно.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ("A", "B", "C", "D")


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# ---------------- A. Суурь ----------------
def base_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a = atr(df)
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12, 24, 72):
        f[f"ret_{k}"] = (c - c.shift(k)) / a
    lr = np.log(c).diff()
    f["vol_24"] = lr.rolling(24).std()
    f["vol_ratio_24_240"] = f["vol_24"] / lr.rolling(240).std()
    f["atr_rel"] = a / c
    f["atr_ratio_14_100"] = a / atr(df, 100)
    f["bar_range_atr"] = (df["high"] - df["low"]) / a
    f["body_atr"] = (c - df["open"]) / a
    f["range24_atr"] = (df["high"].rolling(24).max() - df["low"].rolling(24).min()) / a
    f["vol_tick_ratio"] = df["volume"] / df["volume"].rolling(240).median()
    # Цаг: лааны ХААЛТЫН цаг (таамаглал хийгдэх агшин)
    h = (df.index + pd.Timedelta(hours=1)).hour
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    f["dow"] = df.index.dayofweek
    return f


# ---------------- B. Сонгодог TA ----------------
def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    a = atr(df)
    f = pd.DataFrame(index=df.index)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / dn)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    f["macd_hist_atr"] = (macd - macd.ewm(span=9, adjust=False).mean()) / a
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    for n in (20, 50, 200):
        f[f"dist_ma{n}_atr"] = (c - c.rolling(n).mean()) / a
    return f


# ---------------- C. ICT / SMC ----------------
def _confirmed_swings(df: pd.DataFrame, n: int):
    """Swing high/low-ийн үнийг БАТЛАГДСАН агшинд (i+n) байрлуулна."""
    hi, lo = df["high"], df["low"]
    is_sh = hi == hi.rolling(2 * n + 1, center=True).max()
    is_sl = lo == lo.rolling(2 * n + 1, center=True).min()
    sh = hi.where(is_sh).shift(n)   # i+n цэгт л мэдэгдэнэ
    sl = lo.where(is_sl).shift(n)
    return sh, sl


def ict_features(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    c, hi, lo = df["close"], df["high"], df["low"]
    a = atr(df)
    f = pd.DataFrame(index=df.index)
    sh, sl = _confirmed_swings(df, n)
    last_sh, last_sl = sh.ffill(), sl.ffill()
    prev_sh = sh.dropna().shift(1).reindex(df.index).ffill()
    prev_sl = sl.dropna().shift(1).reindex(df.index).ffill()

    # Бүтэц: HH+HL = +1 (өсөх), LH+LL = -1 (буурах)
    hh, hl = (last_sh > prev_sh), (last_sl > prev_sl)
    f["structure"] = np.where(hh & hl, 1, np.where(~hh & ~hl, -1, 0))
    f["dist_swing_high_atr"] = (last_sh - c) / a
    f["dist_swing_low_atr"] = (c - last_sl) / a
    rng = (last_sh - last_sl).where(lambda x: x > 0)
    f["premium_discount"] = ((c - last_sl) / rng).clip(-0.5, 1.5)   # >0.5 premium, <0.5 discount

    # BOS: хаалт сүүлийн батлагдсан swing-ийг давсан. CHoCH: өмнөх бүтцийн эсрэг BOS.
    bos_up = (c > last_sh) & (c.shift(1) <= last_sh.shift(1))
    bos_dn = (c < last_sl) & (c.shift(1) >= last_sl.shift(1))
    prev_struct = pd.Series(f["structure"], index=df.index).shift(1)
    f["bos_up_recent"] = bos_up.astype(int).rolling(12).max()
    f["bos_dn_recent"] = bos_dn.astype(int).rolling(12).max()
    f["choch_up_recent"] = (bos_up & (prev_struct == -1)).astype(int).rolling(12).max()
    f["choch_dn_recent"] = (bos_dn & (prev_struct == 1)).astype(int).rolling(12).max()

    # Liquidity sweep: swing-ийн цаад талын хөрвөх чадварыг "цуглуулаад" буцаж хаасан
    sweep_hi = (hi > last_sh.shift(1)) & (c < last_sh.shift(1))
    sweep_lo = (lo < last_sl.shift(1)) & (c > last_sl.shift(1))
    f["sweep_high_recent"] = sweep_hi.astype(int).rolling(6).max()
    f["sweep_low_recent"] = sweep_lo.astype(int).rolling(6).max()

    # FVG: 3 лааны загвар; t лааны хаалтад мэдэгдэнэ
    bull = lo > hi.shift(2)
    bear = hi < lo.shift(2)
    f["fvg_bull_24"] = bull.astype(int).rolling(24).sum()
    f["fvg_bear_24"] = bear.astype(int).rolling(24).sum()
    gap = np.where(bull, lo - hi.shift(2), np.where(bear, lo.shift(2) - hi, np.nan))
    f["last_fvg_size_atr"] = pd.Series(gap, index=df.index).ffill() / a
    last_dir = pd.Series(np.where(bull, 1, np.where(bear, -1, np.nan)), index=df.index).ffill()
    f["last_fvg_dir"] = last_dir
    bull_top = pd.Series(np.where(bull, lo, np.nan), index=df.index).ffill()
    bear_bot = pd.Series(np.where(bear, hi, np.nan), index=df.index).ffill()
    f["dist_bull_fvg_atr"] = (c - bull_top) / a
    f["dist_bear_fvg_atr"] = (bear_bot - c) / a

    # Killzone (UTC, хаалтын цаг): Лондон 07–10, Нью-Йорк 12–15
    h = (df.index + pd.Timedelta(hours=1)).hour
    f["kz_london"] = ((h >= 7) & (h < 10)).astype(int)
    f["kz_newyork"] = ((h >= 12) & (h < 15)).astype(int)
    # Өндөр timeframe-ийн чиг: өдрийн (24ц) ба долоо хоногийн (120ц) бүтэц
    f["htf_pos_day"] = (c - lo.rolling(24).min()) / (hi.rolling(24).max() - lo.rolling(24).min())
    f["htf_pos_week"] = (c - lo.rolling(120).min()) / (hi.rolling(120).max() - lo.rolling(120).min())
    return f


# ---------------- D. Мэдээ ----------------
NEWS_COLS = ["ev_count", "ev_surprise_z_sum", "ev_usd", "ev_eur", "hours_since_event",
             "hours_to_next_event", "last_surprise_z"]


def news_block(df: pd.DataFrame) -> pd.DataFrame:
    f = df[NEWS_COLS].copy()
    f["hours_since_event"] = f["hours_since_event"].clip(upper=168)
    f["hours_to_next_event"] = f["hours_to_next_event"].clip(upper=168)
    f["event_within_24h"] = (f["hours_to_next_event"] <= 24).astype(int)
    f["surprise_recent_24"] = df["ev_surprise_z_sum"].fillna(0).rolling(24).sum()
    return f


# Бодит цагт мэдэгдэх (зөвхөн хуваарийн) мэдээний шинжүүд: actual/surprise шаардлагагүй
NEWS_SCHEDULE_COLS = ["ev_count", "ev_usd", "ev_eur", "hours_since_event", "hours_to_next_event", "event_within_24h"]


def build_features(df: pd.DataFrame, groups=GROUPS) -> pd.DataFrame:
    parts = []
    if "A" in groups:
        parts.append(base_features(df))
    if "B" in groups:
        parts.append(ta_features(df))
    if "C" in groups:
        parts.append(ict_features(df))
    if "D" in groups:
        parts.append(news_block(df))
    return pd.concat(parts, axis=1)


def group_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Бүлэг бүрийн баганын нэрс (ablation-д)."""
    small = df.iloc[:400]
    return {
        "A": list(base_features(small).columns),
        "B": list(ta_features(small).columns),
        "C": list(ict_features(small).columns),
        "D": list(news_block(small).columns),
    }
