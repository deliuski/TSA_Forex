"""Бусад зах зээлийн 1 цагийн BID лааг Dukascopy-оос татна (data/raw/markets/<instrument>_<year>.csv).

Сонголт: 2015 оноос өмнөх түүхтэй хэрэгслүүд (dukascopy-node@1.51.1-ийн мета өгөгдлөөр шалгасан):
  gbpusd, usdjpy, usdchf  (2003-аас) — долларын ерөнхий хүч
  xauusd                  (2003-аас) — алт, эрсдэлээс зугтах
  usa500idxusd            (2011-ээс) — S&P 500, эрсдэлийн хандлага
  deuidxeur               (2013-аас) — DAX, Европын хувьцаа
  brentcmdusd             (2010-аас) — газрын тос
DXY (2017-оос), АНУ/Германы бонд (2016/2018-аас), VIX (2022-оос) хэт богино тул ороогүй.

Ажиллуулах:  python scripts/download_markets.py --from-year 2015 --to-year 2025
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DUKASCOPY_NODE = "dukascopy-node@1.51.1"
INSTRUMENTS = ["gbpusd", "usdjpy", "usdchf", "xauusd", "usa500idxusd", "deuidxeur", "brentcmdusd"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/markets")
    ap.add_argument("--only", nargs="*", help="Зөвхөн эдгээрийг (жишээ нь gbpusd xauusd)")
    args = ap.parse_args()
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx олдсонгүй. Node.js 20+ суулгана уу: https://nodejs.org")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    failed = []
    for inst in args.only or INSTRUMENTS:
        for y in range(args.from_year, args.to_year + 1):
            target = out / f"{inst}_{y}.csv"
            if target.exists():
                continue
            cmd = [npx, "-y", DUKASCOPY_NODE, "-i", inst, "-from", f"{y}-01-01", "-to", f"{y + 1}-01-01",
                   "-t", "h1", "-p", "bid", "-v", "-f", "csv", "-dir", str(out), "-fn", target.stem,
                   "--retries", "3", "-ch"]
            print(" ".join(cmd[2:]))
            if subprocess.run(cmd).returncode != 0:
                failed.append(target.name)
    print("Дууслаа:", out.resolve())
    if failed:
        print("Татагдаагүй:", failed, "— дахин ажиллуулахад зөвхөн дутуу файлууд татагдана.")


if __name__ == "__main__":
    main()
