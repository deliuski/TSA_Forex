"""EUR/USD 1 минутын BID лааг Dukascopy-оос сар сараар татна (HistData дууссаны дараах хугацааг нөхөх).

Шаардлага: Node.js 20+. dukascopy-node@1.51.1 (download_dukascopy.py-тай ижил).
Ажиллуулах:  python scripts/download_m1_dukascopy.py --from 2026-01 --to 2026-11
Гаралт: data/raw/dukascopy_m1/eurusd_m1_bid_YYYY-MM.csv. Дууссан сарыг дахин татахгүй (тухайн сарыг бүрэн дахин татна).
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

DUKASCOPY_NODE = "dukascopy-node@1.51.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="2026-01")
    ap.add_argument("--to", dest="end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m"))
    ap.add_argument("--out", default="data/raw/dukascopy_m1")
    a = ap.parse_args()
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx олдсонгүй. Node.js 20+ суулгана уу: https://nodejs.org")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz="UTC")
    for m in pd.period_range(a.start, a.end, freq="M"):
        target = out / f"eurusd_m1_bid_{m}.csv"
        complete = m.end_time.tz_localize("UTC") < now - pd.Timedelta(days=1)
        if target.exists() and complete:
            print(f"[skip] {target}")
            continue
        end = min(m.end_time.tz_localize("UTC") + pd.Timedelta(seconds=1), now + pd.Timedelta(days=1))
        cmd = [npx, "-y", DUKASCOPY_NODE, "-i", "eurusd", "-from", m.start_time.strftime("%Y-%m-%d"),
               "-to", end.strftime("%Y-%m-%d"), "-t", "m1", "-p", "bid", "-v", "-f", "csv",
               "-dir", str(out), "-fn", target.stem, "--retries", "3"]
        print(" ".join(cmd[2:]))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"{m}: татаж чадсангүй\n{r.stderr[-1500:]}")
        n = sum(1 for _ in open(target, encoding="utf-8")) - 1
        print(f"  {target.name}: {n} лаа")


if __name__ == "__main__":
    main()
