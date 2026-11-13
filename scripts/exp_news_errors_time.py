"""Алдааны цаг хугацааны шинжилгээ: surprise-ийн чиглэл таараагүй мэдээнүүд ХЭЗЭЭ гарсан бэ?

exp_news_errors.py-ийн үр дүн (reports/news_errors_events.csv) дээр:
  * он, улирал бүрийн таарц, хөдөлгөөнт 60 үйл явдлын таарц (95% CI)
  * тренд: logit(таарц) ~ он (сүүлийн үед муудсан уу)
  * бөөгнөрөл: Wald–Wolfowitz runs test, хамгийн урт алдааны дараалал vs санамсаргүй сэлгэлт
  * өдрийн цаг (UTC), гариг, онцгой үеүд (COVID, Fed-ийн хүү өсгөлт)
Ажиллуулах: python scripts/exp_news_errors_time.py   (эхлээд exp_news_errors.py)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from statsmodels.sandbox.stats.runs import runstest_1samp  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

REP = ROOT / "reports"


def rate(x):
    n, k = len(x), int(x.sum())
    lo, hi = proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)
    return dict(n=n, hit=round(100 * k / n, 1) if n else None, ci95=[round(100 * lo, 1), round(100 * hi, 1)])


def longest_run(a, val=0):
    best = cur = 0
    for v in a:
        cur = cur + 1 if v == val else 0
        best = max(best, cur)
    return best


def main():
    D = pd.read_csv(REP / "news_errors_events.csv", parse_dates=["bar"]).sort_values("bar").set_index("bar")
    h = D["hit"].to_numpy()
    out = {"overall": rate(D["hit"])}
    out["by_year"] = {str(y): rate(g["hit"]) for y, g in D.groupby(D.index.year)}
    out["by_half_year"] = {f"{y}-H{1 + (m > 6)}": rate(g["hit"])
                           for (y, m), g in D.groupby([D.index.year, D.index.month])}  # түр
    hy = D.index.year.astype(str) + "-H" + np.where(D.index.month <= 6, "1", "2")
    out["by_half_year"] = {k: rate(g["hit"]) for k, g in D.groupby(hy)}
    # Тренд
    yrs = (D.index - D.index[0]).days / 365.25
    fit = sm.Logit(D["hit"], sm.add_constant(pd.Series(np.asarray(yrs), index=D.index, name="years"))).fit(disp=0)
    out["trend_per_year"] = dict(odds_ratio=float(np.exp(fit.params["years"])), p=float(fit.pvalues["years"]))
    first, last = D[D.index < "2020-01-01"], D[D.index >= "2020-01-01"]
    out["2015-2019 vs 2020-2025"] = {"2015-2019": rate(first["hit"]), "2020-2025": rate(last["hit"])}
    # Бөөгнөрөл
    z, p = runstest_1samp(h, cutoff=0.5, correction=True)
    rng = np.random.default_rng(0)
    perm = [longest_run(rng.permutation(h)) for _ in range(2000)]
    out["clustering"] = dict(runs_test_z=float(z), runs_test_p=float(p), longest_miss_streak=longest_run(h),
                             longest_streak_p95_if_random=float(np.percentile(perm, 95)),
                             lag1_autocorr_of_miss=float(pd.Series(1 - h).autocorr(1)))
    # Өдрийн цаг, гариг
    hr = D.index.hour + D.index.minute / 60
    out["by_hour_utc"] = {f"{k:g}": rate(g["hit"]) for k, g in D.groupby(np.round(hr * 2) / 2) if len(g) >= 25}
    out["by_weekday"] = {["Даваа", "Мягмар", "Лхагва", "Пүрэв", "Баасан"][k]: rate(g["hit"]) for k, g in D.groupby(D.index.dayofweek) if k < 5}
    periods = {"COVID 2020-03 – 2020-06": ("2020-03-01", "2020-07-01"),
               "Fed хүү өсгөлт 2022-03 – 2023-07": ("2022-03-01", "2023-08-01"),
               "2015 (ECB QE, хэлбэлзэл өндөр)": ("2015-01-01", "2016-01-01")}
    out["special_periods"] = {}
    for k, (a, b) in periods.items():
        m = (D.index >= pd.Timestamp(a, tz="UTC")) & (D.index < pd.Timestamp(b, tz="UTC"))
        out["special_periods"][k] = {"дотор": rate(D.loc[m, "hit"]), "гадна": rate(D.loc[~m, "hit"])}
    # Жинхэнэ эсрэг хөдөлгөөн (тайлбарлагдаагүй) хэзээ гарсан
    real = D[(D.hit == 0) & (D.absJ >= 2) & (D.conflict == 0) & (D.absS >= 1)]
    out["unexplained_misses_by_year"] = real.groupby(real.index.year).size().to_dict()
    out["events_by_year"] = D.groupby(D.index.year).size().to_dict()
    (REP / "exp_news_errors_time.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_half_year",)}, ensure_ascii=False, indent=1, default=str))

    # График: хөдөлгөөнт 60 үйл явдлын таарц
    roll = D["hit"].rolling(60, center=True, min_periods=40).mean() * 100
    se = np.sqrt(roll / 100 * (1 - roll / 100) / 60) * 100 * 1.96
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(D.index, roll - se, roll + se, alpha=0.2)
    ax.plot(D.index, roll, lw=1.5, label="Хөдөлгөөнт 60 үйл явдлын таарц")
    ax.axhline(100 * D["hit"].mean(), color="grey", ls="--", lw=0.8, label=f"Дундаж {100 * D['hit'].mean():.1f}%")
    ax.axhline(50, color="k", lw=0.6)
    miss = D[D.hit == 0]
    ax.scatter(miss.index, np.full(len(miss), 42), marker="|", s=40, color="#c0504d", alpha=0.6, label="Таараагүй мэдээ")
    for k, (a, b) in periods.items():
        ax.axvspan(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC"), color="orange", alpha=0.12)
    ax.set_ylim(38, 95)
    ax.set_ylabel("Чиглэл таарсан, %")
    ax.set_title("Surprise -> эхний 5 мин: таарц цаг хугацааны дагуу (улбар шар = COVID, Fed хүү өсгөлт, 2015)")
    ax.legend(fontsize=8, loc="lower right", ncol=3)
    fig.tight_layout()
    fig.savefig(REP / "news_errors_time.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
