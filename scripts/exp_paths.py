"""Туршилт: боломжит замуудаас тооцсон "аль түвшинд эхэлж хүрэх" магадлал бодит байдалтай таарах уу?

Test 2023-01 – 2025-04, 6 цаг тутамд. Түвшин: тэгш хэмтэй ±B, B = 2·ATR ба 4·ATR (пип).
Аргууд:
  analog        : нөхцөлтэй түүхэн замууд (өдрийн цаг, мэдээ, хэлбэлзлийн горим), сан = 2022 он хүртэл
  analog_random : ижил сангаас санамсаргүй сонгосон замууд (нөхцөлийн нэмэр болох эсэхийг шалгах)
  gaussian      : санамсаргүй алхалт (сүүлийн 240ц-ийн хэлбэлзэл), тасралтгүй хугацааны хүрэх магадлал
Хэмжүүр: 3 ангиллын log-loss (дээд/доод/аль нь ч биш), "аль нь ч биш"-ийн калибровк, 80% интервалын хамрах хувь.

Ажиллуулах: python scripts/exp_paths.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fxmodel.features import atr  # noqa: E402
from fxmodel.paths import PIP, PathLibrary, barrier_probs, realized_outcome  # noqa: E402

VALID_END = pd.Timestamp("2023-01-01", tz="UTC")
N_PATHS, STEP = 300, 6
REP = ROOT / "reports"


def gaussian_none_table(seed=0, n=20000, steps=400):
    """Тасралтгүй санамсаргүй алхалт: P(max|W_t| < b, t∈[0,1]) — b торын хүснэгт."""
    rng = np.random.default_rng(seed)
    w = np.cumsum(rng.normal(0, np.sqrt(1 / steps), (n, steps)), axis=1)
    m = np.abs(w).max(axis=1)
    grid = np.linspace(0.05, 5, 120)
    return grid, np.array([(m < b).mean() for b in grid])


def logloss3(p, y):
    p = np.clip(p, 1e-4, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), y])))


def main():
    df = pd.read_parquet(ROOT / "data/processed/eurusd_h1_with_news.parquet")
    lib = PathLibrary(24).build(df, df["hours_to_next_event"])
    a_pips = (atr(df) / PIP).to_numpy()
    logvr = np.log((atr(df) / atr(df, 100)).to_numpy())
    hstd = (df["close"].diff() / PIP).rolling(240).std().to_numpy()
    grid, pnone = gaussian_none_table()
    rng = np.random.default_rng(1)
    pool = np.where(lib.index < VALID_END - pd.Timedelta(hours=24))[0]
    test_pos = [i for i in range(len(df)) if df.index[i] >= VALID_END and i % STEP == 0 and i + 25 < len(df)]
    out = {}
    for mult in (2, 4):
        P = {"analog": [], "analog_random": [], "gaussian": []}
        Y, band_hit, days = [], [], []
        for i in test_pos:
            t = df.index[i]
            if (df.index[i + 24] - t) != pd.Timedelta(hours=24) or np.isnan(a_pips[i]) or np.isnan(hstd[i]):
                continue
            B = mult * a_pips[i]
            y = realized_outcome(df, i, B, B)
            if y < 0:
                continue
            hour = (t + pd.Timedelta(hours=1)).hour
            sel = lib.select(hour, logvr[i], df["hours_to_next_event"].iat[i], N_PATHS, before=VALID_END - pd.Timedelta(hours=24))
            for name, s in (("analog", sel), ("analog_random", rng.choice(pool, N_PATHS, replace=False))):
                bp = barrier_probs(lib.high[s] * a_pips[i], lib.low[s] * a_pips[i], B, B)
                P[name].append([bp["up"], bp["down"], bp["none"]])
            b = B / (hstd[i] * np.sqrt(24))
            pn = float(np.interp(b, grid, pnone))
            P["gaussian"].append([(1 - pn) / 2, (1 - pn) / 2, pn])
            Y.append(y)
            days.append(str(t.date()))
            endq = np.quantile(lib.close[sel, -1] * a_pips[i], [0.1, 0.9])
            real_end = (df["close"].iat[i + 24] - df["close"].iat[i]) / PIP
            band_hit.append(endq[0] <= real_end <= endq[1])
        Y = np.array(Y)
        days = np.array(days)
        res = {"n": int(len(Y)), "realized_freq": {k: float(np.mean(Y == j)) for j, k in enumerate(("up", "down", "none"))}}
        for name, p in P.items():
            p = np.array(p)
            res[name] = {"logloss": logloss3(p, Y), "mean_p_none": float(p[:, 2].mean()),
                         "brier_none": float(np.mean((p[:, 2] - (Y == 2)) ** 2))}
            P[name] = p
        # Паарласан bootstrap (өдрөөр): logloss(жишиг) − logloss(analog), >0 = analog сайн
        rng_b = np.random.default_rng(0)
        def ll_vec(p):  # noqa: E306
            p = np.clip(p, 1e-4, 1)
            p = p / p.sum(axis=1, keepdims=True)
            return -np.log(p[np.arange(len(Y)), Y])
        for ref in ("gaussian", "analog_random"):
            d = pd.Series(ll_vec(P[ref]) - ll_vec(P["analog"])).groupby(days).agg(["sum", "count"])
            bs = []
            for _ in range(500):
                smp = d.sample(len(d), replace=True, random_state=rng_b.integers(1 << 31))
                bs.append(smp["sum"].sum() / smp["count"].sum())
            res[f"analog_vs_{ref}_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        res["analog_endpoint_band80_coverage"] = float(np.mean(band_hit))
        # Чиглэл: хүрсэн тохиолдлуудад P(дээд) > P(доод) нь таарсан уу
        hit = Y < 2
        pa = P["analog"]
        res["analog_direction_hit_rate"] = float(np.mean((pa[hit, 0] > pa[hit, 1]) == (Y[hit] == 0)))
        # Калибровк: P(none)-ийн 5 бүлэг
        q = pd.qcut(pa[:, 2], 5, duplicates="drop")
        cal = pd.DataFrame({"p": pa[:, 2], "y": (Y == 2)}).groupby(q, observed=True).mean()
        res["analog_none_calibration"] = [[float(r.p), float(r.y)] for r in cal.itertuples()]
        out[f"B={mult}ATR"] = res
        print(mult, json.dumps({k: v for k, v in res.items() if k != "analog_none_calibration"}, ensure_ascii=False))
    (REP / "exp_paths.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Калибровкийн график
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for k, r in out.items():
        c = np.array(r["analog_none_calibration"])
        ax.plot(c[:, 0], c[:, 1], "o-", label=f"±{k[2:]}")
    ax.set_xlabel("Таамагласан P(аль түвшинд ч хүрэхгүй)")
    ax.set_ylabel("Бодит давтамж")
    ax.set_title("Боломжит замуудын калибровк (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REP / "paths_calibration.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
