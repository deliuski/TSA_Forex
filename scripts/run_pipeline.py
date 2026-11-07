"""Бүх алхмыг нэг дор: унших -> чанар шалгах -> (хөндлөн шалгах) -> мэдээ нэгтгэх -> DB + тайлан.

Жишээ:
  python scripts/run_pipeline.py \
      --dukascopy-dir data/raw/dukascopy \
      --histdata-dir data/raw/histdata \
      --calendar data/raw/calendar/forex_factory.csv --calendar-tz Asia/Tehran
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from fxdata import db, prices, quality, calendar, merge, timing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dukascopy-dir", required=True)
    ap.add_argument("--histdata-dir")
    ap.add_argument("--calendar")
    ap.add_argument("--calendar-tz", default="UTC", help="Календарийн CSV-ийн цагийн бүс (IANA нэр)")
    ap.add_argument("--db", default="data/fx.db")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2026-01-01")
    ap.add_argument("--keep-db", action="store_true",
                    help="Хуучин DB-г устгахгүй (анхдагчаар дахин үүсгэнэ: INSERT OR IGNORE хуучин буруу мөрийг үлдээдэг)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}

    # 1. Ханш
    files = sorted(Path(a.dukascopy_dir).glob("*.csv"))
    if not files:
        sys.exit(f"{a.dukascopy_dir} дотор CSV алга. Эхлээд scripts/download_dukascopy.py ажиллуулна.")
    duka = prices.load_dukascopy_csv(files)
    duka = duka[(duka.index >= pd.Timestamp(a.start, tz="UTC")) & (duka.index < pd.Timestamp(a.end, tz="UTC"))]
    issues = quality.check_prices(duka, source="dukascopy")
    report["dukascopy_rows"] = len(duka)
    report["dukascopy_range"] = [str(duka.index.min()), str(duka.index.max())]
    report["dukascopy_issues"] = quality.summarize(issues).set_index("issue")["count"].to_dict()

    # 2. Хөндлөн шалгалт
    if a.histdata_dir:
        hfiles = sorted(Path(a.histdata_dir).glob("*.zip")) + sorted(Path(a.histdata_dir).glob("*.csv"))
        if hfiles:
            hist = prices.resample_to_h1(prices.load_histdata_m1(hfiles))
            hist = hist[(hist.index >= duka.index.min()) & (hist.index <= duka.index.max())]
            report["cross_check"] = quality.compare_sources(duka, hist)
            hist.to_parquet(out / "eurusd_h1_histdata.parquet")

    clean = quality.clean_prices(duka, issues)
    unclean = duka.index.difference(clean.index)
    clean.to_parquet(out / "eurusd_h1.parquet")
    issues.to_csv(out / "quality_issues.csv", index=False)

    db_path = ROOT / a.db if not Path(a.db).is_absolute() else Path(a.db)
    if db_path.exists() and not a.keep_db:
        db_path.unlink()
    con = db.connect(db_path)
    db.write_candles(con, duka, "dukascopy", unclean=unclean)
    db.write_issues(con, issues)

    # 3. Мэдээ
    if a.calendar:
        ev, log = calendar.load_calendar(a.calendar, a.calendar_tz)
        ev = ev[(ev.ts_utc >= duka.index.min()) & (ev.ts_utc < duka.index.max() + pd.Timedelta(hours=1))]
        report["calendar"] = log | {"in_price_range": len(ev),
                                    "high_impact_in_range": int((ev.impact == "high").sum())}
        ev = db.write_events(con, ev)
        cmap = merge.candle_event_map(clean.index, ev)
        db.write_candle_events(con, cmap)
        report["calendar"]["shifted_to_next_candle"] = int(cmap["shifted"].sum())
        feats = merge.news_features(clean, ev)
        # Календарь дууссаны дараах лаанд мэдээний шинж хүчингүй (hours_since_event зүгээр л өснө)
        cal_end = ev["ts_utc"].max().ceil("D")
        with_news = clean.join(feats)
        with_news = with_news[with_news.index < cal_end]
        with_news.to_parquet(out / "eurusd_h1_with_news.parquet")
        report["calendar"]["with_news_range"] = [str(with_news.index.min()), str(with_news.index.max())]
        report["calendar"]["candles_after_calendar_end_dropped"] = int((clean.index >= cal_end).sum())
        # Spike-уудын хэд нь өндөр нөлөөтэй мэдээний цагт таарч байна вэ? (бодит хөдөлгөөн эсэх)
        sp = issues.loc[issues.issue == "spike", "ts"]
        hi = ev[ev.impact == "high"]
        hi_hours = set(hi["ts_utc"].dt.floor("h"))
        report["spikes_on_high_news_hours"] = int(sum(t in hi_hours for t in sp))
        report["spikes_total"] = int(len(sp))
        # Мэдээний цаг зөв лаанд буусан эсэх (event study): оргил t=0 дээр байх ёстой
        es = timing.event_study(clean, hi)
        es.to_csv(out / "news_timing_check.csv")
        timing.plot_event_study({"Өндөр нөлөөтэй мэдээ (time_source-аар)": es}, out / "news_timing_check.png")
        report["news_timing_check"] = es.round(2).to_dict(orient="index")

    (out / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
