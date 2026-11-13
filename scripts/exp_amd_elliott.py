"""Туршилт: ICT AMD (Power of 3 / Judas swing) ба Elliott-ийн "5 давалгааны дараа эргэнэ" таамаглал.

AMD: Ази (00–07 UTC) = accumulation, Лондон (07–12) = manipulation (Азийн хүрээний нэг талыг эвдэх),
     өдрийн үлдсэн хэсэг (–21 UTC) = distribution. Таамаглал: Лондон эхлээд эвдсэн талын ЭСРЭГ тийш өдөр хаагдана.
     Null: эвдсэн цэгээс санамсаргүй алхалтаар (тухайн өдрийн хэлбэлзлээр) Азийн дундаас цааш эсрэг талд хаах магадлал.
Elliott: H1 дээрх батлагдсан swing-ийн дараалсан HH (эсвэл LL) тоо k -> ирэх 24 цагийн өгөөж (ATR, трендийн чиглэлд).
     Таамаглал: k=5-ийн дараа эргэнэ (дундаж < 0, үргэлжлэх магадлал < 50%).
Ажиллуулах: python scripts/exp_amd_elliott.py   (data/processed/eurusd_m5.parquet шаардлагатай — exp_m5.py үүсгэнэ)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.stats import norm  # noqa: E402

from fxmodel.features import _confirmed_swings, atr  # noqa: E402
from fxmodel.intraday import load_m5_cached  # noqa: E402


def amd(m5):
    rows = []
    for day, g in m5.groupby(m5.index.normalize()):
        a, ld = g[g.index.hour < 7], g[(g.index.hour >= 7) & (g.index.hour < 12)]
        rest = g[(g.index.hour >= 7) & (g.index.hour < 21)]
        if len(a) < 70 or len(ld) < 50 or len(rest) < 150:
            continue
        AH, AL = a.high.max(), a.low.min()
        mid = (AH + AL) / 2
        upb, dnb = ld.index[ld.high > AH], ld.index[ld.low < AL]
        if len(upb) and (not len(dnb) or upb[0] < dnb[0]):
            f, t0, lvl = "up", upb[0], AH
        elif len(dnb):
            f, t0, lvl = "down", dnb[0], AL
        else:
            continue
        after = rest[rest.index >= t0]
        close = after.close.iloc[-1]
        sig = np.log(rest.close).diff().dropna().std() * np.sqrt(len(after))
        p_rw = norm.cdf(-abs(np.log(lvl) - np.log(mid)) / sig)
        rows.append(dict(first=f, rev=(close < mid) if f == "up" else (close > mid), p_rw=p_rw,
                         asia=(AH - AL) / 1e-4, rest=(rest.high.max() - rest.low.min()) / 1e-4))
    D = pd.DataFrame(rows)
    out = {}
    for f in ("up", "down", "all"):
        s = D if f == "all" else D[D["first"] == f]
        z = (s.rev.sum() - s.p_rw.sum()) / np.sqrt((s.p_rw * (1 - s.p_rw)).sum())
        out[f] = dict(n=int(len(s)), reversal_pct=100 * float(s.rev.mean()), random_walk_pct=100 * float(s.p_rw.mean()),
                      z=float(z), p=float(2 * (1 - norm.cdf(abs(z)))))
    out["asia_vs_rest_range_spearman"] = float(stats.spearmanr(D.asia, D.rest).correlation)
    return out


def elliott(h1):
    sh, sl = _confirmed_swings(h1, 3)
    fut = (h1.close.shift(-24) - h1.close) / atr(h1)
    res = []
    for kind, s, sign in (("up", sh, 1), ("down", sl, -1)):
        pts = s.dropna()
        seq = (np.sign(pts.diff()) == sign).astype(int)
        run = seq.groupby((seq == 0).cumsum()).cumsum()
        df = pd.DataFrame({"run": run, "fut": fut.reindex(pts.index) * sign}).dropna()
        for k in range(1, 8):
            x = df.fut[df.run == k] if k < 7 else df.fut[df.run >= 7]
            res.append(dict(trend=kind, k=k if k < 7 else "7+", n=int(len(x)), mean_atr=float(x.mean()),
                            continue_pct=100 * float((x > 0).mean()), p=float(stats.ttest_1samp(x, 0).pvalue)))
    return res


def main():
    m5 = load_m5_cached(ROOT)
    m5 = m5[m5.index.dayofweek < 5]
    h1 = pd.read_parquet(ROOT / "data/processed/eurusd_h1.parquet")
    out = {"amd": amd(m5), "elliott": elliott(h1), "n_tests_elliott": 14, "bonferroni_alpha": 0.05 / 14}
    (ROOT / "reports" / "exp_amd_elliott.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["amd"], ensure_ascii=False, indent=1))
    print(pd.DataFrame(out["elliott"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
