"""Алдааны шинжилгээ: surprise-ийн чиглэлтэй эхний 5 минутын хөдөлгөөн ЯАГААД таарахгүй байна вэ (~30%)?

Үйл явдал: |S_max| >= 0.5, эхний 5 минутын хөдөлгөөн J != 0 (2015 – 2025-04, M5).
Хүчин зүйлс:
  surprise-ийн хэмжээ |S|, хариу үйлдлийн хэмжээ |J|, нэг цагт гарсан мэдээнүүдийн surprise зөрчилтэй эсэх,
  мэдээний төрөл, валют, мэдээний өмнөх 30 минутын хөдөлгөөн (surprise-ийн чиглэлд), хэлбэлзлийн горим, он.
Арга: хүчин зүйл бүрээр таарцын хувь (Wilson 95% CI) + бүх хүчин зүйлийг зэрэг оруулсан логистик регресс.
"Тодорхой" тохиолдлууд (том surprise, зөрчилгүй, бодит хариу үйлдэл) дээр таарц хэд хүрэхийг харуулна.

Ажиллуулах: python scripts/exp_news_errors.py
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
import statsmodels.api as sm  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from fxdata import calendar  # noqa: E402
from fxmodel.intraday import load_m5_cached  # noqa: E402
from fxmodel.news_events import build_events, usual_effect  # noqa: E402

REP = ROOT / "reports"


def rate(df, col="hit"):
    n, k = len(df), int(df[col].sum())
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return dict(n=n, hit=round(100 * k / n, 1) if n else None, ci95=[round(100 * lo, 1), round(100 * hi, 1)])


def main():
    m5 = load_m5_cached(ROOT)
    raw = ROOT / "data/raw/calendar/forex_factory.csv"
    ev, _ = calendar.load_calendar(raw, "Asia/Tehran")
    E = build_events(m5, ev, usual_effect(raw))
    D = E[(E["S_max"].abs() >= 0.5) & (E["J"] != 0)].copy()
    D["hit"] = (np.sign(D["J"]) == np.sign(D["S_max"])).astype(int)
    D["absS"], D["absJ"] = D["S_max"].abs(), D["J"].abs()
    D["conflict"] = ((D["n_pos"] > 0) & (D["n_neg"] > 0)).astype(int)
    D["pre_with_S"] = np.sign(D["S_max"]) * D["pre_ret30"]            # >0: зах зээл урьдчилан surprise-ийн чиглэлд хөдөлсөн
    D["vol_regime"] = D["pre_r24h"] / D["pre_r24h"].rolling(60, min_periods=10).median()
    D["year"] = D.index.year
    out = {"overall": rate(D)}

    def by(col, bins=None, labels=None):
        g = pd.cut(D[col], bins, labels=labels) if bins is not None else D[col]
        return {str(k): rate(v) for k, v in D.groupby(g, observed=True)}

    out["|S| (surprise хэмжээ, z)"] = by("absS", [0.5, 1, 1.5, 2.5, 99], ["0.5–1", "1–1.5", "1.5–2.5", ">2.5"])
    out["|J| (эхний 5 мин хөдөлгөөн, пип)"] = by("absJ", [0, 2, 5, 10, 999], ["<2", "2–5", "5–10", ">10"])
    out["Зөрчилтэй surprise (нэг цагт эсрэг чиглэлийн мэдээ)"] = {"үгүй": rate(D[D.conflict == 0]), "тийм": rate(D[D.conflict == 1])}
    out["Мэдээний өмнөх 30 мин (surprise-ийн чиглэлд, пип)"] = by("pre_with_S", [-999, -5, 0, 5, 999],
                                                                   ["эсрэг >5", "эсрэг 0–5", "дагуу 0–5", "дагуу >5"])
    out["Валют"] = {k: rate(v) for k, v in D.groupby("key_currency")}
    out["Хэлбэлзлийн горим (24ц хүрээ / 60 үйл явдлын медиан)"] = by("vol_regime", [0, 0.8, 1.25, 99], ["тайван", "ердийн", "өндөр"])
    out["Он"] = {str(k): rate(v) for k, v in D.groupby("year")}
    types = D["key_title"].value_counts()
    out["Мэдээний төрөл (≥20 тохиолдол)"] = dict(sorted(((t, rate(D[D.key_title == t])) for t in types[types >= 20].index),
                                                         key=lambda kv: -kv[1]["hit"]))

    # Логистик регресс: бүх хүчин зүйлийг зэрэг (odds ratio, p)
    X = pd.DataFrame({"log_absS": np.log(D["absS"]), "log_absJ": np.log(D["absJ"]), "conflict": D["conflict"],
                      "pre_with_S_10pips": D["pre_with_S"] / 10, "usd": (D["key_currency"] == "USD").astype(int),
                      "log_vol_regime": np.log(D["vol_regime"].fillna(1)), "n_events": D["n_events"]})
    fit = sm.Logit(D["hit"], sm.add_constant(X)).fit(disp=0)
    out["logit"] = {k: dict(odds_ratio=round(float(np.exp(fit.params[k])), 3), p=float(fit.pvalues[k])) for k in X.columns}

    # "Тодорхой" тохиолдлууд
    clear = D[(D.absS >= 1) & (D.conflict == 0) & (D.absJ >= 5)]
    out["Тодорхой тохиолдол: |S|≥1, зөрчилгүй, |J|≥5 пип"] = rate(clear)
    out["Үлдсэн алдааны бүтэц"] = {
        "нийт алдаа": int((D.hit == 0).sum()),
        "|J|<2 пип (бараг хариу үйлдэлгүй)": int(((D.hit == 0) & (D.absJ < 2)).sum()),
        "зөрчилтэй surprise": int(((D.hit == 0) & (D.conflict == 1)).sum()),
        "|S|<1 (сул surprise)": int(((D.hit == 0) & (D.absS < 1)).sum()),
        "эдгээрийн аль нь ч биш (жинхэнэ эсрэг хөдөлгөөн)": int(((D.hit == 0) & (D.absJ >= 2) & (D.conflict == 0) & (D.absS >= 1)).sum()),
    }
    real = D[(D.hit == 0) & (D.absJ >= 2) & (D.conflict == 0) & (D.absS >= 1)]
    out["Жинхэнэ эсрэг хөдөлгөөний мэдээний төрөл"] = real["key_title"].value_counts().head(8).to_dict()
    (REP / "exp_news_errors.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    D.to_csv(REP / "news_errors_events.csv")
    print(json.dumps(out, ensure_ascii=False, indent=1))

    # График: |S| ба |J|-ээр таарц
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
    for ax, key in zip(axes, ["|S| (surprise хэмжээ, z)", "|J| (эхний 5 мин хөдөлгөөн, пип)",
                              "Мэдээний өмнөх 30 мин (surprise-ийн чиглэлд, пип)"]):
        d = out[key]
        ks = list(d)
        hits = [d[k]["hit"] for k in ks]
        err = [[h - d[k]["ci95"][0] for k, h in zip(ks, hits)], [d[k]["ci95"][1] - h for k, h in zip(ks, hits)]]
        ax.bar(ks, hits, yerr=err, capsize=4, color="#3b6fb6")
        ax.axhline(50, color="k", lw=0.7, ls="--")
        for i, k in enumerate(ks):
            ax.text(i, 3, f"n={d[k]['n']}", ha="center", color="white", fontsize=8)
        ax.set_title(key, fontsize=10)
    axes[0].set_ylabel("Чиглэл таарсан, % (95% CI)")
    axes[0].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(REP / "news_errors.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
