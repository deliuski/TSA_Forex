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

# HistData: "Eastern Standard Time WITHOUT daylight saving" = жилийн турш UTC-5.
# "America/New_York" ашиглавал зуны цагийн үед 1 цагаар зөрнө — энэ бол түгээмэл алдаа.
HISTDATA_UTC_OFFSET_HOURS = 5


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
    df["ts"] = (local + pd.Timedelta(hours=HISTDATA_UTC_OFFSET_HOURS)).dt.tz_localize("UTC")
    return df[["ts", *OHLCV]]


def load_histdata_m1(paths: list[Path] | Path) -> pd.DataFrame:
    """HistData Generic ASCII M1 (zip эсвэл csv). Цагийг UTC руу хөрвүүлнэ (+5 цаг)."""
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
