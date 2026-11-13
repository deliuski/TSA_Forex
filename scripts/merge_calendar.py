"""Шинэ ForexFactory календарийг (2025-04-07-ноос хойш) хуучин файлтай нэгтгэнэ.

1) Скрейпер (таны компьютер дээр, нэг удаа):
     git clone https://github.com/ehsanrs2/forexfactory-scraper.git ..\\forexfactory-scraper
     cd ..\\forexfactory-scraper
     python -m pip install -r requirements.txt
     python -m src.forexfactory.main --start 2025-04-06 --end 2026-11-08 --impact high,medium ^
            --keep-currencies USD EUR --csv ff_2025_2026.csv --tz UTC
2) Нэгтгэх:
     python scripts/merge_calendar.py --new ..\\forexfactory-scraper\\ff_2025_2026.csv

Хуудасны цагийн бүсийг тогтмол хуваарьтай мэдээнүүдээр автоматаар тодорхойлно (файлын offset-д итгэхгүй).
90%-иас бага таарвал зогсоно. Хуучин файлыг data/raw/calendar/forex_factory_2007_2025-04.csv нэрээр нөөцөлнө.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fxdata import calendar_update as cu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True, help="Скрейперийн CSV (DateTime, Currency, Impact, Event, Actual, ...)")
    ap.add_argument("--old", default=str(ROOT / "data/raw/calendar/forex_factory.csv"))
    ap.add_argument("--page-tz", help="Хуудасны цагийн бүсийг гараар заах (анхдагч: автоматаар таах)")
    ap.add_argument("--min-share", type=float, default=0.9)
    a = ap.parse_args()

    old_path = Path(a.old)
    backup = old_path.with_name("forex_factory_2007_2025-04.csv")
    if not backup.exists():
        shutil.copy(old_path, backup)
        print(f"Нөөц: {backup}")
    old = pd.read_csv(backup, dtype=str, keep_default_na=False)       # үргэлж анхны файлаас эхэлнэ (давтан ажиллуулж болно)
    new = pd.read_csv(a.new, dtype=str, keep_default_na=False)
    need = {"DateTime", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"}
    if not need <= set(new.columns):
        sys.exit(f"{a.new}: {sorted(need - set(new.columns))} багана алга. forexfactory-scraper-ийн гаралт мөн үү?")
    new = new[new["Currency"].str.upper().isin(["USD", "EUR"])]
    print(f"Шинэ файл: {len(new)} USD/EUR мөр, {new['DateTime'].str[:10].min()} – {new['DateTime'].str[:10].max()}")

    probe = pd.DataFrame({"currency": new["Currency"].str.upper(), "title": new["Event"].str.strip(), "wall": cu._wall(new["DateTime"])})
    inf = cu.infer_page_tz(probe)
    print(f"Хуудасны цагийн бүс: {inf['tz']} ({inf['matches']}/{inf['n']} хуваарьт мэдээ таарсан, {100 * inf['share']:.1f}%). "
          f"Бусад: {inf['ranking']}")
    tz = a.page_tz or inf["tz"]
    if not a.page_tz and inf["share"] < a.min_share:
        sys.exit("Цагийн бүс тодорхойгүй байна (< 90% таарсан). Скрейперийн гаралтыг шалгах эсвэл --page-tz заана уу.")

    conv = cu.to_old_format(new, tz, cu.detail_map_from_old(old))
    merged, info = cu.merge(old, conv)
    info |= dict(page_tz=tz, tz_inference=inf)
    merged.to_csv(old_path, index=False)
    (ROOT / "data/raw/calendar/merge_report.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str),
                                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in info.items() if k != "tz_inference"}, ensure_ascii=False, indent=1, default=str))
    print(f"Хадгалсан: {old_path}. Дараа нь: python scripts/run_pipeline.py ... --end 2026-12-01 (README-г харна уу)")


if __name__ == "__main__":
    main()
