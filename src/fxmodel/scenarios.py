"""Ханшийн "хувилбар" (scenario) буюу ирээдүйн замын хэлбэрүүд.

Лаа бүрийн хувьд дараагийн H цагийн хаалтын замыг ATR-аар хэвийн болгоно:
    path_t[k] = (close[t+k] - close[t]) / ATR[t],   k = 1..H
Ингэснээр "хэдэн пип" биш "одоогийн хэлбэлзэлтэй харьцуулахад хэр хол" гэдгийг харьцуулна.
Замыг 6 хувилбарт тодорхой дүрмээр (classify_paths) оноож, загвар (LightGBM) нь одоогийн
нөхцөлөөс хувилбар бүрийн магадлалыг таамаглана.

Яагаад k-means биш вэ: туршихад k-means (K=6) замуудыг зөвхөн эцсийн утгаар нь
(их уналт ... их өсөлт) хуваасан. Ханш санамсаргүй алхалтад ойр тул замын хамгийн их
хэлбэлзэл нь эцсийн утгаас үүсдэг. Range, "унаад өснө" гэх мэт хэлбэрүүд гарч ирээгүй
тул тайлбарлахад амар, ICT-ийн ойлголттой нийцэх дүрмийн хувилбарыг сонгосон.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr


def future_paths(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    c = df["close"].to_numpy()
    a = atr(df).to_numpy()
    n = len(c)
    out = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        out[: n - k, k - 1] = (c[k:] - c[: n - k]) / a[: n - k]
    # Амралтын өдрийг дамнасан замыг хасна (лааны индекс тасралтгүй биш)
    gap = pd.Series(df.index).diff(horizon).shift(-horizon).dt.total_seconds().to_numpy() / 3600
    out[gap > horizon + 2] = np.nan
    return pd.DataFrame(out, index=df.index, columns=[f"k{k}" for k in range(1, horizon + 1)])


NAMES = [
    "Хажуу тийш (range)",
    "Тогтвортой өсөлт",
    "Тогтвортой уналт",
    "Эхлээд унаад буцаж өснө",
    "Эхлээд өсөөд буцаж унана",
]


def classify_paths(paths: np.ndarray, s: float) -> np.ndarray:
    """Замыг 5 хувилбарын аль нэгэнд оноох дүрэм. s = "мэдэгдэхүйц хөдөлгөөн"-ий босго (ATR).

    up = замын хамгийн дээд цэг, dn = хамгийн доод цэг (эерэг тоо), end = эцсийн утга.
      max(up, dn) < s                      -> 0 range
      up >= s ба dn >= s (хоёр тийш)       -> |end| < s/2: аль туйл эхэлснээр 3 эсвэл 4
                                              end > 0: доод цэг эхэлсэн бол 3 (унаад өснө), үгүй бол 1
                                              end < 0: дээд цэг эхэлсэн бол 4 (өсөөд унана), үгүй бол 2
    (Хоёр тийш савлаад 0 орчим хаах тохиолдол сургалтад 0.3% байсан тул 3/4-т нэгтгэв.)
      зөвхөн up >= s                       -> өсөлтийн тал хувиас илүүг хадгалсан бол 1, үгүй бол 4
      зөвхөн dn >= s                       -> уналтын тал хувиас илүүг хадгалсан бол 2, үгүй бол 3
    3 ба 4 нь ICT-ийн "liquidity sweep -> эргэлт" санаатай таарна.
    """
    up = np.maximum(paths.max(axis=1), 0)
    dn = np.maximum(-paths.min(axis=1), 0)
    end = paths[:, -1]
    t_max, t_min = paths.argmax(axis=1), paths.argmin(axis=1)
    lab = np.zeros(len(paths), dtype=int)
    both = (up >= s) & (dn >= s)
    only_up = (up >= s) & ~both
    only_dn = (dn >= s) & ~both
    mid = both & (np.abs(end) < s / 2)
    lab[mid] = np.where(t_min[mid] < t_max[mid], 3, 4)
    lab[both & (end >= s / 2)] = np.where(t_min[both & (end >= s / 2)] < t_max[both & (end >= s / 2)], 3, 1)
    lab[both & (end <= -s / 2)] = np.where(t_max[both & (end <= -s / 2)] < t_min[both & (end <= -s / 2)], 4, 2)
    lab[only_up] = np.where(end[only_up] >= 0.5 * up[only_up], 1, 4)
    lab[only_dn] = np.where(-end[only_dn] >= 0.5 * dn[only_dn], 2, 3)
    return lab


class ScenarioModel:
    """Дүрэмд суурилсан хувилбарууд. Босго s-ийг сургалтын өгөгдлөөс (квантилаар) тогтооно."""

    def __init__(self, range_quantile: float = 0.30):
        self.range_quantile = range_quantile
        self.names = NAMES
        self.k = len(NAMES)

    def fit(self, paths: pd.DataFrame):
        x = paths.dropna().to_numpy()
        m = np.maximum(x.max(axis=1), -x.min(axis=1))
        self.s = float(np.quantile(m, self.range_quantile))
        lab = classify_paths(x, self.s)
        self.base_rates = np.bincount(lab, minlength=self.k) / len(lab)
        self.centroids = np.vstack([x[lab == i].mean(axis=0) for i in range(self.k)])
        self.q20 = np.vstack([np.quantile(x[lab == i], 0.2, axis=0) for i in range(self.k)])
        self.q80 = np.vstack([np.quantile(x[lab == i], 0.8, axis=0) for i in range(self.k)])
        return self

    def assign(self, paths: pd.DataFrame) -> np.ndarray:
        return classify_paths(paths.to_numpy(), self.s)
