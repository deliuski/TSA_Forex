"""Мэдээний төрөл бүрийн профайл (бодит цагийн системд): хүлээгдэх 60 минутын хүрээ, ердийнхөөс хэд дахин том,
surprise -> эхний 5 минутын чиглэлийн таарц. Гаралт: models/news_profile.json

Өмнө нь: python scripts/exp_news.py && python scripts/exp_news_errors.py (reports/*events.csv үүсгэнэ)
Ажиллуулах: python scripts/build_news_profile.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fxmodel.news_events import news_profile, usual_effect  # noqa: E402


def main():
    E = pd.read_csv(ROOT / "reports/news_events.csv")
    D = pd.read_csv(ROOT / "reports/news_errors_events.csv")
    eff = usual_effect(ROOT / "data/raw/calendar/forex_factory.csv")
    prof = news_profile(E, D, eff)
    out = dict(profile=prof, effect=[[c, t, v] for (c, t), v in eff.items()],
               note="2015 – 2025-04, EUR/USD M5. r60 = мэдээнээс хойших 60 минутын хүрээ (пип).")
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models/news_profile.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    top = sorted(((k, v) for k, v in prof.items() if k != "_global" and v["n"] >= 10), key=lambda kv: -kv[1]["mult"])
    print(f"{len(prof) - 1} төрөл. Хамгийн их нөлөөтэй:")
    for k, v in top[:10]:
        print(f"  {k:35s} ×{v['mult']:.2f}  медиан {v['r60_med']:.0f} пип (10–90%: {v['r60_q10']:.0f}–{v['r60_q90']:.0f}), "
              f"чиглэл {100 * v['hit5']:.0f}%  n={v['n']}")
    print("Хамгийн бага:", ", ".join(f"{k} ×{v['mult']:.2f}" for k, v in top[-4:]))


if __name__ == "__main__":
    main()
