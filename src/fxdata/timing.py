"""Мэдээний цаг зөв лаанд буусан эсэхийг ханшийн хариу үйлдлээр шалгах (event study).

Санаа: өндөр нөлөөтэй мэдээ (NFP, CPI, FOMC...) гарсан цагийн лааны хүрээ (high-low)
тухайн цагийн ердийн хүрээнээс хэд дахин том байдаг. Мэдээг зөв лаанд холбосон бол
"мэдээний лаа / ердийн лаа" харьцаа 0 шилжилт (t=0) дээр хамгийн өндөр гарах ёстой.
Хэрэв t-1 эсвэл t+1 дээр өндөр гарвал цагийн бүс/offset 1 цагаар зөрсөн гэсэн үг.

"Ердийн хүрээ" = ижил цагийн (hour-of-day) сүүлийн 20 арилжааны өдрийн медиан. Өдрийн
хэлбэлзлийн хэв маяг (Лондон/NY session) нөлөөллийг ингэж хасна.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def normal_range(candles: pd.DataFrame, lookback_days: int = 20) -> pd.Series:
    """Лаа бүрийн хувьд ижил цагийн өмнөх lookback_days өдрийн медиан хүрээ (пип биш, үнэ)."""
    rng = candles["high"] - candles["low"]
    hour = candles.index.hour
    out = pd.Series(np.nan, index=candles.index)
    for h in range(24):
        s = rng[hour == h]
        out[s.index] = s.rolling(lookback_days, min_periods=5).median().shift(1)
    return out


def event_study(candles: pd.DataFrame, events: pd.DataFrame, shifts=(-2, -1, 0, 1, 2),
                by: str = "time_source") -> pd.DataFrame:
    """Мэдээ бүрийн лааны шилжилт бүрт range / normal_range харьцааны МЕДИАНыг буцаана."""
    rng = candles["high"] - candles["low"]
    ratio = (rng / normal_range(candles)).replace([np.inf, -np.inf], np.nan)
    ev = events.copy()
    ev["candle_ts"] = ev["ts_utc"].dt.floor("h")
    ev = ev[ev["candle_ts"].isin(candles.index)]
    pos = candles.index.get_indexer(ev["candle_ts"])
    rows = {}
    for k in shifts:
        p = pos + k
        ok = (p >= 0) & (p < len(candles))
        vals = np.full(len(ev), np.nan)
        vals[ok] = ratio.to_numpy()[p[ok]]
        rows[k] = vals
    df = pd.DataFrame(rows, index=ev.index)
    df[by] = ev[by].to_numpy() if by in ev else "all"
    out = df.groupby(by)[list(shifts)].median()
    out["n"] = df.groupby(by).size()
    out["peak_shift"] = out[list(shifts)].idxmax(axis=1)
    return out


def plot_event_study(tables: dict[str, pd.DataFrame], path, shifts=(-2, -1, 0, 1, 2)):
    """Хэд хэдэн хувилбарын (жишээ нь засвараас өмнө/хойш) event study графикийг хадгална."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 4), sharey=True, squeeze=False)
    for ax, (name, t) in zip(axes[0], tables.items()):
        for label, row in t.iterrows():
            ax.plot(list(shifts), row[list(shifts)].astype(float), marker="o", label=f"{label} (n={int(row['n'])})")
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.axhline(1, color="grey", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("Мэдээний лаанаас шилжилт (цаг)")
        ax.set_xticks(list(shifts))
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("Лааны хүрээ / ердийн хүрээ (медиан)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
