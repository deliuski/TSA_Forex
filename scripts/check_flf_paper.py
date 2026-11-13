"""FLF-LSTM (Ahmed et al., 2020, Applied Soft Computing 97:106780) судалгааны нийтэлсэн өгөгдөл,
таамаглалыг энгийн baseline-тай харьцуулна.

Өгөгдөл: https://github.com/slab-itu/forex_flf_lstm  ->  "Data and Code.zip" задлаад Data folder-ийн замыг өгнө.
Ажиллуулах: python scripts/check_flf_paper.py "C:/path/to/Data"

Тэдний кодын дагуу (tagData, percentage=60): оролт = d[6046:-1] (одоогийн лаа), зорилт = d[6047:] (дараагийн лаа).
Ground_truth.csv нь d[6046:] буюу ОРОЛТТОЙ ижил байдгийг анхаар.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "Data")
d = np.loadtxt(folder / "EURUSDmicro_ H4_CandleData.txt", delimiter=",")
sz = int(np.ceil(len(d) * 0.6))
X, Y = d[sz - 1:-1], d[sz:]
mae = lambda p: np.abs(p[: len(Y)] - Y).mean(axis=0) * 1e4  # noqa: E731
rows = {}
for f in sorted(folder.glob("*_*.csv")):
    if f.stem == "Ground_truth":
        continue
    p = pd.read_csv(f, index_col=0).to_numpy()[: len(Y)]
    rows[f"{f.stem}  (High<Low: {100 * (p[:, 1] < p[:, 2]).mean():.0f}%)"] = mae(p)
tr = d[:sz]
c = X[:, 3]
rows["NAIVE-1: дараагийн OHLC = одоогийн OHLC"] = mae(X)
rows["NAIVE-2: O=C, C=C, H/L = C ± train-ийн дундаж сүүдэр"] = mae(
    np.c_[c, c + (tr[:, 1] - tr[:, 3]).mean(), c - (tr[:, 3] - tr[:, 2]).mean(), c])
rows["Нийтлэлийн Table 1: FLF-LSTM (ReLU)"] = np.array([15.49, 18.85, 29.90, 23.32])
rows["Нийтлэлийн Table 1: OHLC-LSTM (ReLU)"] = np.array([19.17, 20.28, 32.18, 26.07])
t = pd.DataFrame(rows, index=["Open", "High", "Low", "Close"]).T
t["дундаж"] = t.mean(axis=1)
print("MAE, пип (test: 4030 H4 лаа, 2017–2018)")
print(t.round(1).to_string())
