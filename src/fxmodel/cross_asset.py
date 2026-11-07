"""Бусад зах зээлийн шинжүүд (бүлэг X). data/raw/markets/<instrument>_<year>.csv байвал ашиглана.

Look-ahead-аас сэргийлэх: бусад хэрэгслийг EUR/USD-ийн лааны цагтай ЗӨВХӨН өнгөрсөн утгаар
(asof/ffill, хамгийн ихдээ 3 цаг) тааруулна. Индекс CFD-ууд шөнө хаалттай байдаг тул сүүлийн
мэдэгдэж буй үнийг ашиглана, ирээдүйн үнийг хэзээ ч ашиглахгүй.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fxdata import prices
from .features import atr

INSTRUMENTS = ["gbpusd", "usdjpy", "usdchf", "xauusd", "usa500idxusd", "deuidxeur", "brentcmdusd"]
# Доллар өсөхөд өсдөг чиглэл (+1) эсвэл буурдаг (-1)
USD_SIGN = {"gbpusd": -1, "usdjpy": 1, "usdchf": 1}


def load_markets(folder: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for inst in INSTRUMENTS:
        files = sorted(Path(folder).glob(f"{inst}_*.csv"))
        if files:
            d = prices.load_dukascopy_csv(files)
            out[inst] = d[~d.index.duplicated()]
    return out


def _align(s: pd.Series, index: pd.DatetimeIndex, limit_h: int = 3) -> pd.Series:
    """Зөвхөн өнгөрсөн утгаар (t-ээс өмнөх эсвэл t-тэй тэнцүү хамгийн сүүлийн лаа) тааруулна."""
    full = s.reindex(s.index.union(index)).ffill(limit=limit_h)
    return full.reindex(index)


def cross_features(eur: pd.DataFrame, mk: dict[str, pd.DataFrame]) -> pd.DataFrame:
    f = pd.DataFrame(index=eur.index)
    eur_r = np.log(eur["close"]).diff()
    usd = []
    for inst, d in mk.items():
        a = atr(d)
        c = d["close"]
        for k in (1, 6, 24):
            f[f"x_{inst}_ret{k}"] = _align((c - c.shift(k)) / a, eur.index)
        f[f"x_{inst}_volratio"] = _align(a / atr(d, 100), eur.index)
        if inst in USD_SIGN:
            usd.append(USD_SIGN[inst] * f[[f"x_{inst}_ret6", f"x_{inst}_ret24"]])
        r = _align(np.log(c).diff(), eur.index)
        if inst in ("gbpusd", "xauusd", "usa500idxusd"):
            f[f"x_corr_eur_{inst}_120"] = eur_r.rolling(120, min_periods=60).corr(r)
    if usd:
        u = sum(x.to_numpy() for x in usd) / len(usd)
        f["x_usd_strength6"] = u[:, 0]
        f["x_usd_strength24"] = u[:, 1]
    return f
