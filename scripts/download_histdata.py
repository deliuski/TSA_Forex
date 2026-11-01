"""Хөндлөн шалгалтад зориулж HistData-аас EUR/USD M1 (BID) татна. Цаг нь UTC-5 (зуны цаггүй).

Ажиллуулах:  python scripts/download_histdata.py --from-year 2015 --to-year 2025
"""
import argparse
from pathlib import Path

from histdata import download_hist_data as dl
from histdata.api import Platform as P, TimeFrame as TF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/histdata")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for y in range(a.from_year, a.to_year + 1):
        if list(out.glob(f"*EURUSD*_{y}.zip")):
            print(f"[skip] {y}")
            continue
        print(dl(year=str(y), month=None, pair="eurusd", platform=P.GENERIC_ASCII,
                 time_frame=TF.ONE_MINUTE, output_directory=str(out)))


if __name__ == "__main__":
    main()
