"""Ирэх 24 цагийн ханшийн хүрээний (high − low, пип) таамаглал.

Яагаад: хувилбарын загварын үнэлгээгээр загвар ЧИГЛЭЛИЙГ таамаглаж чаддаггүй (AUC 0.50) ч
ХӨДӨЛГӨӨНИЙ ХЭМЖЭЭГ таамаглаж чаддаг (AUC 0.63) нь харагдсан. Тиймээс сигнал байгаа газарт
шууд төвлөрнө. Хүрээний таамаглалыг stop-ийн зай, байрлалын хэмжээ, арилжаа хийх эсэхийг
шийдэхэд ашиглаж болно.

Baseline: HAR (Corsi, 2009) — өдрийн, долоо хоногийн, сарын хэлбэлзлийн шугаман хослол.
Хэлбэлзлийн таамаглалд хамгийн өргөн хэрэглэгддэг, давахад хэцүү жишиг.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PIP = 1e-4


def future_range(df: pd.DataFrame, horizon: int = 24) -> pd.Series:
    """t-ийн дараах horizon лааны max(high) − min(low), пипээр. Амралтын өдрийг дамнавал NaN."""
    hi = df["high"][::-1].rolling(horizon, min_periods=horizon).max()[::-1].shift(-1)
    lo = df["low"][::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    r = (hi - lo) / PIP
    gap = pd.Series(df.index).diff(horizon).shift(-horizon).dt.total_seconds().to_numpy() / 3600
    r[gap > horizon + 2] = np.nan
    return r


def past_range(df: pd.DataFrame, horizon: int = 24) -> pd.Series:
    """t хүртэлх (t-г оруулаад) сүүлийн horizon лааны хүрээ, пипээр."""
    return (df["high"].rolling(horizon).max() - df["low"].rolling(horizon).min()) / PIP


def har_features(df: pd.DataFrame) -> pd.DataFrame:
    """HAR шинжүүд log-оор: сүүлийн өдөр, 5 өдрийн дундаж, 22 өдрийн дундаж (өдөр бүр 24 цагаар алхана)."""
    lr = np.log(past_range(df).clip(lower=1))
    f = pd.DataFrame(index=df.index)
    f["har_d"] = lr
    f["har_w"] = sum(lr.shift(24 * k) for k in range(5)) / 5
    f["har_m"] = sum(lr.shift(24 * k) for k in range(22)) / 22
    h = (df.index + pd.Timedelta(hours=1)).hour
    f["har_hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["har_hour_cos"] = np.cos(2 * np.pi * h / 24)
    return f


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))
