"""Мэдээний агшны чиглэлийг (surprise -> эхний минутууд) АРИЛЖААНД ашиглаж болох уу?

Асуулт: surprise-ийн чиглэл эхний 5 минутад 70% таардаг. Гэвч энэ хөдөлгөөн хэдхэн секундэд болдог.
Бодит арилжаачин мэдээ гарснаас d минутын дараа (d = 1, 2, 3, 5) орж, мэдээний үеийн өргөн spread төлөөд
ашиг олох уу?

Өгөгдөл: HistData M1 (BID), өндөр нөлөөтэй USD/EUR мэдээ, |S_max| >= 0.5 (2015 – 2025-04).
Орох: мэдээ гарсан минутаас d минутын дараах үнэ (t0+d минутын нээлт), surprise-ийн чиглэлд.
Гарах: t0 + {5, 15, 30, 60} минут.
Зардал: 1, 2, 3 пип (round-trip; мэдээний үеийн spread + slippage).
Хуваалт: дүрмийг (d, гарах хугацаа) 2015–2021 дээр сонгоод 2023+ дээр нэг удаа шалгана.
Шүүлтүүр: зөрчилтэй surprise-ийг алгасах (алдааны шинжилгээнд батлагдсан цорын ганц дүрэм).
Нэмэлт асуулт: эхний d минутын хөдөлгөөн surprise-ийн чиглэлтэй таарсан үед л орвол?

Ажиллуулах: python scripts/exp_news_latency.py
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

from fxdata import calendar  # noqa: E402
from fxmodel.intraday import load_m5_cached  # noqa: E402
from fxmodel.intraday import load_m1_all  # noqa: E402
from fxmodel.splits import TEST_END, TRAIN_END, VALID_END  # noqa: E402
from fxmodel.news_events import build_events, usual_effect  # noqa: E402

REP = ROOT / "reports"
PIP = 1e-4
DELAYS = [0, 1, 2, 3, 5]          # 0 = мэдээ гарсан минутын нээлт (бодит бус, дээд хязгаар)
EXITS = [5, 15, 30, 60]
COSTS = [1.0, 2.0, 3.0]
RNG = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 5:
        return [np.nan, np.nan]
    m = RNG.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [round(float(np.percentile(m, 2.5)), 2), round(float(np.percentile(m, 97.5)), 2)]


def main():
    m1 = load_m1_all(ROOT / "data/raw/histdata")
    m1 = m1[~m1.index.duplicated()].sort_index()
    m1.index = m1.index.as_unit("ns")
    m5 = load_m5_cached(ROOT)
    raw = ROOT / "data/raw/calendar/forex_factory.csv"
    ev, _ = calendar.load_calendar(raw, "Asia/Tehran")
    E = build_events(m5, ev, usual_effect(raw))
    E = E[E["S_max"].abs() >= 0.5].copy()
    E["conflict"] = ((E["n_pos"] > 0) & (E["n_neg"] > 0)).astype(int)
    E["dir"] = np.sign(E["S_max"])

    # M1 дээрх үнэ: t0+k минутын нээлт (k = 0..60), тасралтгүй байх ёстой
    idx = m1.index
    op = m1["open"].to_numpy()
    t0 = pd.DatetimeIndex(E.index).as_unit("ns")
    pos = idx.get_indexer(t0)
    keep, P = [], []
    for i, p in enumerate(pos):
        if p < 0 or p + 61 >= len(idx) or (idx[p + 60] - idx[p]) != pd.Timedelta(minutes=60):
            continue
        keep.append(i)
        P.append(op[p:p + 61])
    E = E.iloc[keep].copy()
    P = np.vstack(P)                                        # (n, 61) үнэ
    E["year"] = E.index.year
    print(f"M1 дээр бүрэн үйл явдал: {len(E)}")

    t = E.index
    split = np.where(t < TRAIN_END, "train", np.where((t >= VALID_END) & (t < TEST_END), "test",
                                                      np.where(t >= TEST_END, "forward", "valid")))
    out = {"n_events": int(len(E)), "grid": {}}
    rows = []
    for flt in ["бүгд", "зөрчилгүй"]:
        mask_f = np.ones(len(E), bool) if flt == "бүгд" else (E["conflict"].to_numpy() == 0)
        for d in DELAYS:
            for x in EXITS:
                if x <= d:
                    continue
                g = E["dir"].to_numpy() * (P[:, x] - P[:, d]) / PIP   # зардлын өмнөх пип
                for sp in ["train", "test", "forward"]:
                    m = mask_f & (split == sp)
                    r = g[m]
                    rows.append(dict(filter=flt, delay=d, exit=x, split=sp, n=int(m.sum()),
                                     hit=round(100 * float((r > 0).mean()), 1),
                                     gross=round(float(r.mean()), 2), gross_ci=boot_ci(r),
                                     **{f"net_{c:g}": round(float(r.mean() - c), 2) for c in COSTS}))
    T = pd.DataFrame(rows)
    T.to_csv(REP / "news_latency_grid.csv", index=False)

    # Дүрмийг train дээр сонгох: 2 пипийн зардалтай, d >= 1 (бодитой)
    cand = T[(T.split == "train") & (T.delay >= 1)].sort_values("net_2", ascending=False)
    best = cand.iloc[0]
    tb = T[(T.split == "test") & (T["filter"] == best["filter"]) & (T.delay == best.delay) & (T.exit == best.exit)].iloc[0]
    out["best_on_train"] = {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v)) for k, v in best.items()}
    out["same_rule_on_test"] = {k: (v if not isinstance(v, (np.integer, np.floating)) else float(v)) for k, v in tb.items()}

    # Эхний d минутын хөдөлгөөн surprise-тэй таарсан үед л орох (батлалт)
    conf = []
    for d in [1, 2, 3]:
        early = E["dir"].to_numpy() * (P[:, d] - P[:, 0]) / PIP
        for x in [15, 30, 60]:
            g = E["dir"].to_numpy() * (P[:, x] - P[:, d]) / PIP
            for sp in ["train", "test", "forward"]:
                for lab, mm in [("эхний хөдөлгөөн таарсан", early > 0), ("таараагүй", early < 0)]:
                    m = mm & (split == sp) & (E["conflict"].to_numpy() == 0)
                    r = g[m]
                    conf.append(dict(delay=d, exit=x, split=sp, case=lab, n=int(m.sum()),
                                     hit=round(100 * float((r > 0).mean()), 1), gross=round(float(r.mean()), 2),
                                     gross_ci=boot_ci(r), net_2=round(float(r.mean() - 2), 2)))
    C = pd.DataFrame(conf)
    C.to_csv(REP / "news_latency_confirm.csv", index=False)

    # Хөдөлгөөний хэдэн хувь нь эхний минутад болдог вэ (surprise-ийн чиглэлд, 5 минутын хөдөлгөөнтэй харьцуулахад)
    move = E["dir"].to_numpy()[:, None] * (P - P[:, [0]]) / PIP
    med = {f"+{k}м": round(float(np.median(move[:, k])), 2) for k in [1, 2, 3, 5, 15, 30, 60]}
    mean = {f"+{k}м": round(float(np.mean(move[:, k])), 2) for k in [1, 2, 3, 5, 15, 30, 60]}
    out["cum_move_in_surprise_dir_pips"] = {"median": med, "mean": mean}
    out["grid_summary_test"] = T[(T.split == "test") & (T["filter"] == "зөрчилгүй")][
        ["delay", "exit", "n", "hit", "gross", "gross_ci", "net_1", "net_2", "net_3"]].to_dict("records")
    out["confirm_test"] = C[C.split == "test"].to_dict("records")
    (REP / "exp_news_latency.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # График: surprise-ийн чиглэлд дундаж хуримтлагдсан хөдөлгөөн (train/test)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for sp, col in [("train", "#888"), ("test", "#1f77b4")]:
        m = (split == sp) & (E["conflict"].to_numpy() == 0)
        ax[0].plot(range(61), move[m].mean(0), color=col, label=f"{sp} (n={m.sum()})")
    ax[0].axhline(0, color="k", lw=0.6)
    for d in [1, 2, 3, 5]:
        ax[0].axvline(d, color="#ddd", lw=0.8, zorder=0)
    ax[0].set(title="Surprise-ийн чиглэлд дундаж хөдөлгөөн (зөрчилгүй)", xlabel="Мэдээнээс хойших минут", ylabel="пип")
    ax[0].legend()
    tt = T[(T.split == "test") & (T["filter"] == "зөрчилгүй")]
    for d, mk in zip(DELAYS, "o^sDv"):
        s = tt[tt.delay == d]
        ax[1].plot(s.exit, s.net_2, marker=mk, label=f"орох +{d}м")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set(title="Test 2023+: 2 пип зардлын дараах дундаж ашиг", xlabel="Гарах минут", ylabel="пип / арилжаа")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REP / "news_latency.png", dpi=130)

    print(json.dumps({k: out[k] for k in ["best_on_train", "same_rule_on_test", "cum_move_in_surprise_dir_pips"]},
                     ensure_ascii=False, indent=1, default=str))
    print(T[(T.split == "test") & (T["filter"] == "зөрчилгүй")].to_string())
    print(C.to_string())


if __name__ == "__main__":
    main()
