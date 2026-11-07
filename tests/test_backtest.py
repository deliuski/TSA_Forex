"""Backtest хөдөлгүүрийн тест (гараар тооцсон тохиолдлууд)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fxmodel import backtest as bt  # noqa: E402


def frame(highs, lows, closes, start="2024-03-04 00:00"):
    idx = pd.date_range(start, periods=len(closes), freq="h", tz="UTC")
    c = np.array(closes, float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": highs, "low": lows, "close": c, "volume": 1.0}, index=idx)


def flat_then(n_flat, tail):
    """ATR ≈ 10 пип байх тогтвортой хэсэг, дараа нь тест хийх хөдөлгөөн."""
    c = [1.1000] * n_flat
    h = [x + 0.0005 for x in c]
    l = [x - 0.0005 for x in c]
    for (hh, ll, cc) in tail:
        h.append(hh); l.append(ll); c.append(cc)
    return frame(h, l, c)


def test_long_hits_take_profit_with_costs():
    df = flat_then(40, [(1.1005, 1.0995, 1.1000), (1.1030, 1.0999, 1.1025), (1.1025, 1.1020, 1.1022)])
    sig = pd.Series(0, index=df.index)
    sig.iloc[40] = 1                                     # дохио 40-д, орох 41-ийн нээлт = 1.1000
    t = bt.simulate(df, sig, cfg=bt.BarrierConfig(tp_atr=1.5, sl_atr=1.5, cost_pips=1.0))
    assert len(t) == 1 and t.reason.iloc[0] == "tp"
    a = t.atr_pips.iloc[0]
    assert abs(t.pnl_pips.iloc[0] - (1.5 * a - 1.0)) < 1e-6


def test_same_bar_tp_and_sl_counts_as_loss():
    df = flat_then(40, [(1.1005, 1.0995, 1.1000), (1.1050, 1.0950, 1.1000), (1.1005, 1.0995, 1.1000)])
    sig = pd.Series(0, index=df.index)
    sig.iloc[40] = 1
    t = bt.simulate(df, sig, cfg=bt.BarrierConfig(cost_pips=0))
    assert t.reason.iloc[0] == "sl" and t.pnl_pips.iloc[0] < 0


def test_take_filter_and_news_cost():
    df = flat_then(40, [(1.1005, 1.0995, 1.1000)] * 30)
    sig = pd.Series(0, index=df.index)
    sig.iloc[40] = -1
    assert bt.simulate(df, sig, take=pd.Series(False, index=df.index)).empty
    t = bt.simulate(df, sig, cfg=bt.BarrierConfig(cost_pips=1.0, news_extra_pips=1.5), news_hours={df.index[41]})
    assert t.reason.iloc[0] == "time" and abs(t.pnl_pips.iloc[0] + 2.5) < 1e-6


def test_dsr_penalizes_many_trials():
    sr, n = 0.08, 600
    psr = bt.probabilistic_sharpe(sr, n, 0.0, 3.0)
    dsr = bt.deflated_sharpe(sr, n, 0.0, 3.0, list(np.random.default_rng(0).normal(0, 0.05, 50)))
    assert 0.5 < psr < 1 and dsr < psr
