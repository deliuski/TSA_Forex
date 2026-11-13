"""Боломжит замууд (analog paths): одоогийн нөхцөлтэй төстэй түүхэн тохиолдлуудын дараах 24 цагийн БОДИТ замууд.

Санаа: замуудыг зохиомлоор үүсгэхийн оронд түүхээс авна. Ингэснээр сессийн хэв маяг, мэдээний огцом
хөдөлгөөн, "fat tail" зэрэг бодит шинж чанарыг агуулна.
  Төстэй байдлын шалгуур: өдрийн ижил цаг (хаалтын цаг), ирэх 24 цагт өндөр нөлөөтэй мэдээ байгаа эсэх,
  хэлбэлзлийн горим (ATR14/ATR100), дараагийн мэдээ хүртэлх хугацаа.
  Масштаб: түүхэн зам нь өөрийн эхлэлийн ATR-аар хэвийн болсон, одоогийн ATR-аар үржүүлнэ. Хэлбэлзлийн
  загварын таамаг (24ц хүрээ) өгвөл өргөнийг түүнд тааруулна.

Түвшин (барьер): +U ба −D пип. Зам бүрийн H1 high/low-оор аль түвшинд ЭХЛЭЖ хүрснийг тооцно. Нэг лаанд
хоёуланд нь хүрвэл хагасаар хуваана. Ингэж P(дээд), P(доод), P(аль нь ч биш)-ийг гаргана.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr

PIP = 1e-4


class PathLibrary:
    def __init__(self, horizon: int = 24):
        self.h = horizon

    def build(self, df: pd.DataFrame, hours_to_next_event: pd.Series | None = None):
        h = self.h
        a = atr(df).to_numpy()
        c, hi, lo = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
        n = len(c)
        span = pd.Series(df.index).shift(-h).to_numpy() - df.index.to_numpy()
        ok = (span == np.timedelta64(h, "h")) & ~np.isnan(a)
        ok[n - h:] = False
        ix = np.where(ok)[0]
        steps = ix[:, None] + np.arange(1, h + 1)[None, :]
        base, aa = c[ix][:, None], a[ix][:, None]
        self.close = (c[steps] - base) / aa
        self.high = (hi[steps] - base) / aa
        self.low = (lo[steps] - base) / aa
        self.index = df.index[ix]
        self.hour = (self.index + pd.Timedelta(hours=1)).hour.to_numpy()
        self.logvr = np.log((atr(df) / atr(df, 100)).to_numpy()[ix])
        htn = (hours_to_next_event.reindex(df.index).to_numpy()[ix] if hours_to_next_event is not None
               else np.full(len(ix), np.nan))
        self.htn = np.nan_to_num(np.clip(htn, 0, 48), nan=48.0)
        self.news = self.htn <= 24
        self.range_atr = self.high.max(axis=1) - self.low.min(axis=1)
        return self

    def select(self, hour: int, logvr: float, htn: float, n: int = 200, before=None) -> np.ndarray:
        """Хамгийн төстэй n түүхэн тохиолдлын индекс."""
        htn = 48.0 if htn is None or np.isnan(htn) else float(np.clip(htn, 0, 48))
        m = (self.hour == hour) & (self.news == (htn <= 24)) & ~np.isnan(self.logvr)
        if before is not None:
            m &= self.index < before
        cand = np.where(m)[0]
        if len(cand) < n:                                  # хангалтгүй бол цагийн нөхцөлийг сулруулна
            cand = np.where(~np.isnan(self.logvr) & ((self.index < before) if before is not None else True))[0]
        sd = np.nanstd(self.logvr) or 1.0
        dist = np.abs(self.logvr[cand] - logvr) / sd + np.abs(self.htn[cand] - htn) / 24.0
        return cand[np.argsort(dist)[:n]]

    def scale_factor(self, sel: np.ndarray, atr_pips: float, range_q50_pips: float | None) -> float:
        if not range_q50_pips:
            return 1.0
        return float(range_q50_pips / (np.median(self.range_atr[sel]) * atr_pips))


def barrier_probs(high: np.ndarray, low: np.ndarray, up: float, down: float) -> dict:
    """high/low: (n_paths, h) пипээр, эхлэлээс. Аль түвшинд эхэлж хүрснийг тоолно."""
    H = high.shape[1]
    hu = high >= up
    hd = low <= -down
    fu = np.where(hu.any(1), hu.argmax(1), H + 1)
    fd = np.where(hd.any(1), hd.argmax(1), H + 1)
    p_up = np.mean(fu < fd) + 0.5 * np.mean((fu == fd) & (fu <= H))
    p_dn = np.mean(fd < fu) + 0.5 * np.mean((fu == fd) & (fu <= H))
    return {"up": float(p_up), "down": float(p_dn), "none": float(1 - p_up - p_dn),
            "outcome": np.where(fu < fd, 0, np.where(fd < fu, 1, np.where(fu <= H, 0, 2)))}


def realized_outcome(df: pd.DataFrame, i: int, up: float, down: float, h: int = 24) -> int:
    """Бодит үр дүн: 0 = дээд түвшинд эхэлж хүрсэн, 1 = доод, 2 = аль нь ч биш (нэг лаанд хоёулаа бол 0.5 биш — хасна)."""
    c0 = df["close"].iat[i]
    hh = (df["high"].to_numpy()[i + 1:i + 1 + h] - c0) / PIP
    ll = (df["low"].to_numpy()[i + 1:i + 1 + h] - c0) / PIP
    u, d = hh >= up, ll <= -down
    fu = u.argmax() if u.any() else h + 1
    fd = d.argmax() if d.any() else h + 1
    if fu == fd and fu <= h:
        return -1
    return 0 if fu < fd else (1 if fd < fu else 2)
