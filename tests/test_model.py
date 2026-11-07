"""Загварын давхаргын тест (синтетик өгөгдөл)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fxdata import quality  # noqa: E402
from fxmodel.features import build_features  # noqa: E402
from fxmodel.scenarios import ScenarioModel, classify_paths, future_paths  # noqa: E402


def synth(n_days=60, seed=1):
    idx = quality.expected_trading_hours(pd.Timestamp("2024-01-01", tz="UTC"),
                                         pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=n_days))
    rng = np.random.default_rng(seed)
    c = 1.08 + np.cumsum(rng.normal(0, 0.0008, len(idx)))
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) + rng.uniform(1e-4, 8e-4, len(idx)),
                       "low": np.minimum(o, c) - rng.uniform(1e-4, 8e-4, len(idx)), "close": c,
                       "volume": rng.uniform(1e3, 5e3, len(idx))}, index=pd.DatetimeIndex(idx, name="ts"))
    for col in ["ev_count", "ev_usd", "ev_eur"]:
        df[col] = 0
    for col in ["ev_surprise_z_sum", "hours_since_event", "hours_to_next_event", "last_surprise_z"]:
        df[col] = np.nan
    return df


def test_features_have_no_lookahead():
    df = synth()
    full = build_features(df)
    for cut in (500, 800, len(df) - 7):
        part = build_features(df.iloc[:cut])
        a, b = full.iloc[cut - 30:cut], part.iloc[-30:]
        same = ((a - b).abs() < 1e-12) | (a.isna() & b.isna())
        assert same.all().all(), f"look-ahead: {same.columns[~same.all()].tolist()}"


def test_classify_paths_rules():
    s = 2.0
    t = np.linspace(0, 1, 24)
    paths = np.vstack([
        0.3 * np.sin(6 * t),            # 0 range
        4 * t,                          # 1 өсөлт
        -4 * t,                         # 2 уналт
        -3 * np.sin(np.pi * t) + t,     # 3 унаад буцаж өснө
        3 * np.sin(np.pi * t) - t,      # 4 өсөөд буцаж унана
    ])
    assert classify_paths(paths, s).tolist() == [0, 1, 2, 3, 4]


def test_future_paths_skip_weekend_and_model_fits():
    df = synth()
    p = future_paths(df, 24)
    fri = df.index[(df.index.dayofweek == 4) & (df.index.hour == 12)][0]
    assert p.loc[fri].isna().all()          # амралтын өдрийг дамнасан зам хасагдана
    sm = ScenarioModel().fit(p.dropna())
    assert abs(sm.base_rates.sum() - 1) < 1e-9 and sm.centroids.shape == (5, 24)


def test_cross_features_no_lookahead_and_alignment():
    from fxmodel.cross_asset import cross_features
    eur = synth(seed=3)
    other = synth(seed=4)
    other = other[other.index.hour % 5 != 0]          # өөр цагийн хуваарьтай (зарим цаг алга)
    full = cross_features(eur, {"gbpusd": other, "xauusd": other * 1500})
    for cut in (400, 700):
        part = cross_features(eur.iloc[:cut], {"gbpusd": other[other.index <= eur.index[cut - 1]],
                                               "xauusd": other[other.index <= eur.index[cut - 1]] * 1500})
        a, b = full.iloc[cut - 20:cut], part.iloc[-20:]
        same = ((a - b).abs() < 1e-9) | (a.isna() & b.isna())
        assert same.all().all()
    assert "x_usd_strength24" in full and "x_corr_eur_gbpusd_120" in full


def test_future_range_matches_manual():
    from fxmodel.volatility import future_range, past_range
    df = synth(seed=5)
    r = future_range(df, 24)
    i = 200
    if not np.isnan(r.iloc[i]):
        w = df.iloc[i + 1:i + 25]
        assert abs(r.iloc[i] - (w.high.max() - w.low.min()) / 1e-4) < 1e-6
    p = past_range(df, 24)
    w = df.iloc[i - 23:i + 1]
    assert abs(p.iloc[i] - (w.high.max() - w.low.min()) / 1e-4) < 1e-6
