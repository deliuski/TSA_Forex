"""EUR/USD 1 цагийн BID лааг Dukascopy-оос жил жилээр татна.

Шаардлага: Node.js 20+ (https://nodejs.org). Python-оос dukascopy-node CLI-г дуудна.
Яагаад өөрөө downloader бичихгүй вэ: Dukascopy 2026 онд өгөгдлийн API-гаа өөрчилж, AWS WAF
хамгаалалт нэмсэн. dukascopy-node үүнийг дагаж шинэчлэгддэг (v1.50, 1.51), тиймээс хувилбарыг
тогтоож (pin) ашиглах нь давтагдах боломжтой (reproducible).

Ажиллуулах:  python scripts/download_dukascopy.py --from-year 2015 --to-year 2025
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DUKASCOPY_NODE = "dukascopy-node@1.51.1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-year", type=int, default=2015)
    ap.add_argument("--to-year", type=int, default=2025)
    ap.add_argument("--out", default="data/raw/dukascopy")
    args = ap.parse_args()

    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        sys.exit("npx олдсонгүй. Node.js 20+ суулгана уу: https://nodejs.org")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for y in range(args.from_year, args.to_year + 1):
        target = out / f"eurusd_h1_bid_{y}.csv"
        if target.exists():
            print(f"[skip] {target}")
            continue
        cmd = [npx, "-y", DUKASCOPY_NODE,
               "-i", "eurusd", "-from", f"{y}-01-01", "-to", f"{y + 1}-01-01",
               "-t", "h1", "-p", "bid", "-v", "-f", "csv",
               "-dir", str(out), "-fn", target.stem,
               "--retries", "3", "-ch"]
        print(" ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"{y} он татахад алдаа гарлаа (код {r.returncode}). "
                     "403 гарвал https://www.dukascopy.com/swiss/english/marketwatch/historical/ "
                     "хуудас браузерт ажиллаж байгаа эсэхийг шалгаад dukascopy-node-ийн хувилбарыг шинэчилнэ.")
    print("Дууслаа:", out.resolve())


if __name__ == "__main__":
    main()
