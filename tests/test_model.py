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


def test_intraday_htf_and_news_no_lookahead():
    from fxmodel.intraday import htf_context, news_minutes, future_range_bars
    idx = pd.date_range("2024-03-04 00:00", periods=12 * 24 * 12, freq="5min", tz="UTC")
    idx = idx[idx.dayofweek < 5]
    rng = np.random.default_rng(7)
    c = 1.08 + np.cumsum(rng.normal(0, 2e-4, len(idx)))
    o = np.r_[c[0], c[:-1]]
    m5 = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1e-4, "low": np.minimum(o, c) - 1e-4,
                       "close": c, "n_m1": 5}, index=idx)
    full = htf_context(m5)
    cut = 1500
    part = htf_context(m5.iloc[:cut])
    a, b = full.iloc[cut - 30:cut], part.iloc[-30:]
    assert (((a - b).abs() < 1e-9) | (a.isna() & b.isna())).all().all()
    # H1 10:00 лаа 11:00-д хаагдана: 10:50 M5 (хаалт 10:55) нь 09:00 H1-ийг харна, 10:55 M5 (хаалт 11:00) 10:00-ийг харна
    ev = pd.DataFrame({"ts_utc": [pd.Timestamp("2024-03-05 12:30", tz="UTC")], "impact": ["high"], "surprise_z": [2.0]})
    nm = news_minutes(m5, ev)
    assert nm.loc[pd.Timestamp("2024-03-05 12:25", tz="UTC"), "min_to_next_event"] == 0      # хаалт 12:30 = мэдээ
    assert np.isnan(nm.loc[pd.Timestamp("2024-03-05 12:25", tz="UTC"), "last_surprise_z_abs"])
    assert nm.loc[pd.Timestamp("2024-03-05 12:30", tz="UTC"), "last_surprise_z_abs"] == 2.0
    r = future_range_bars(m5, 12)
    assert r.isna().sum() > 0 and r.notna().sum() > 0


def test_barrier_probs_and_library():
    from fxmodel.paths import PathLibrary, barrier_probs
    hi = np.array([[1, 5, 9], [1, 2, 3], [12, 1, 1], [1, 1, 1]], float)
    lo = np.array([[-1, -1, -1], [-11, -1, -1], [-1, -1, -1], [-10, -1, -1]], float)
    # зам 0: дээд (k=2), зам 1: доод, зам 2: дээд, зам 3: доод (k=0, high 1 < 8)
    p = barrier_probs(hi, lo, up=8, down=10)
    assert abs(p["up"] - 0.5) < 1e-9 and abs(p["down"] - 0.5) < 1e-9 and abs(p["none"]) < 1e-9
    df = synth(seed=9)
    lib = PathLibrary(24).build(df)
    assert lib.close.shape[1] == 24 and len(lib.index) > 100
    sel = lib.select(hour=10, logvr=0.0, htn=float("nan"), n=50)
    assert len(sel) == 50
    s = lib.scale_factor(sel, atr_pips=10.0, range_q50_pips=float(np.median(lib.range_atr[sel]) * 10.0))
    assert abs(s - 1.0) < 1e-9


def test_news_postmortem_categories_and_no_lookahead():
    import numpy as np
    import pandas as pd
    from fxmodel.news_events import parse_value, postmortem, surprise_sign

    assert parse_value("228K") == 228000 and parse_value("<0.10%") == 0.10 and parse_value("-1.2B") == -1.2e9
    assert np.isnan(parse_value("")) and np.isnan(parse_value(None))
    eff = {("USD", "NFP"): 1, ("USD", "UR"): -1, ("EUR", "CPI"): 1}
    assert surprise_sign("USD", "NFP", "228K", "137K", eff) == -1      # сайн USD -> EUR/USD доош
    assert surprise_sign("EUR", "CPI", "0.5%", "0.3%", eff) == 1
    assert np.isnan(surprise_sign("USD", "NFP", None, "137K", eff))

    t0 = pd.Timestamp("2025-04-04 12:30", tz="UTC")
    idx = pd.date_range(t0 - pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2), freq="1min")
    px = np.full(len(idx), 1.1000)
    k = idx.get_loc(t0)
    px[k + 1:] = 1.0970                                                  # мэдээний дараа 30 пип доош
    m1 = pd.DataFrame({"open": px, "high": px + 1e-5, "low": px - 1e-5, "close": px}, index=idx)
    prof = {"NFP": dict(r60_q10=10, r60_med=20, r60_q90=25, hit5=0.77), "UR": dict(r60_q10=10, r60_med=20, r60_q90=25, hit5=0.5),
            "_global": dict(r60_q10=5, r60_med=10, r60_q90=20, hit5=0.7)}
    ev = pd.DataFrame(dict(ts_utc=[t0], currency=["USD"], impact=["high"], title=["NFP"], actual=["228K"], forecast=["137K"]))
    it = postmortem(m1, ev, prof, eff, t0 + pd.Timedelta(hours=2))[0]
    assert it["hit"] is True and it["size"] == "хүлээгдсэнээс том" and it["J"] < -25
    # Зөрчилтэй surprise: NFP сайн (доош), UR муу (дээш)
    ev2 = pd.concat([ev, ev.assign(title="UR", actual="4.2%", forecast="4.1%")])
    it2 = postmortem(m1, ev2, prof, eff, t0 + pd.Timedelta(hours=2))[0]
    assert it2["hit"] is None and it2["reason"].startswith("зөрчилтэй")
    # Ирээдүйн өгөгдөл ашиглахгүй: "одоо" = мэдээнээс 20 минутын дараа -> дуусаагүй гэж тэмдэглэнэ
    it3 = postmortem(m1[m1.index < t0 + pd.Timedelta(minutes=20)], ev, prof, eff, t0 + pd.Timedelta(minutes=20))[0]
    assert it3["complete"] is False
