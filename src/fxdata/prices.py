"""Ханшийн өгөгдөл унших, нэг хэлбэрт (UTC, 1 цаг, BID) оруулах.

Бүх DataFrame-ийн хэлбэр:
    index  : ts (UTC, лааны НЭЭЛТИЙН цаг; 12:00 лаа = [12:00, 13:00))
    columns: open, high, low, close, volume
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# HistData-ийн баримт бичигт "EST without DST" (тогтмол UTC-5) гэж бичсэн ч энэ төслийн
# 2015–2025 оны файлуудыг Dukascopy (UTC)-тай тулгахад бодит байдал өөр гарсан:
#   тогтмол +5 цаг:      өвөл 0.01 пип, зун 5.0 пип зөрүү (зун 1 цагаар хоцорно)
#   America/New_York:    өвөл ба зун хоёулаа ~0 пип
# Dukascopy-ийн UTC зөв болохыг NFP-ээр баталсан: зун 12:00 UTC, өвөл 13:00 UTC лаа хамгийн том.
# Тиймээс HistData-г Нью-Йоркийн зуны цагтай бүсээр хөрвүүлнэ.
HISTDATA_TZ = "America/New_York"


def load_dukascopy_csv(paths: list[Path] | Path) -> pd.DataFrame:
    """dukascopy-node-ийн CSV (timestamp=ms UTC, open, high, low, close, volume)."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = [c.strip().lower() for c in df.columns]
        if "timestamp" not in df.columns:
            raise ValueError(f"{p}: 'timestamp' багана алга (dukascopy-node -f csv гаралт биш?)")
        ts = df["timestamp"]
        if pd.api.types.is_numeric_dtype(ts):
            df["ts"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["ts"] = pd.to_datetime(ts, utc=True)
        if df["ts"].min() < pd.Timestamp("2000-01-01", tz="UTC"):
            raise ValueError(f"{p}: timestamp 2000 оноос өмнө гарлаа — нэгж (ms/s) буруу байж магадгүй")
        if "volume" not in df.columns:
            df["volume"] = float("nan")
        frames.append(df[["ts", *OHLCV]])
    out = pd.concat(frames, ignore_index=True)
    return _finalize(out)


def _read_histdata_frame(fh) -> pd.DataFrame:
    df = pd.read_csv(
        fh, sep=";", header=None,
        names=["dt", "open", "high", "low", "close", "volume"],
        dtype={"dt": str},
    )
    local = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S")
    # Зах зээл Ням 17:00 NY-д нээгддэг тул зуны цагийн шилжилтийн цаг (Ням 02:00) хаалттай байна.
    df["ts"] = local.dt.tz_localize(HISTDATA_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    df = df[df["ts"].notna()]
    return df[["ts", *OHLCV]]


def load_histdata_m1(paths: list[Path] | Path) -> pd.DataFrame:
    """HistData Generic ASCII M1 (zip эсвэл csv). Цагийг America/New_York-оос UTC руу хөрвүүлнэ."""
    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for name in z.namelist():
                    if name.lower().endswith(".csv"):
                        with z.open(name) as fh:
                            frames.append(_read_histdata_frame(io.TextIOWrapper(fh)))
        else:
            frames.append(_read_histdata_frame(p))
    return _finalize(pd.concat(frames, ignore_index=True))


def resample_to_h1(m1: pd.DataFrame) -> pd.DataFrame:
    """1 минутын лааг 1 цагийн лаа болгоно (label/closed = left)."""
    agg = m1.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open"])


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ts")
    df = df.set_index("ts")
    df.index.name = "ts"
    return df[OHLCV].astype(float)
