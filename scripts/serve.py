"""Бодит цагийн чартын сервер (FastAPI + TradingView Lightweight Charts).

Нэг командаар: цаг бүр таамаглал гаргаж (predict_live), вэб хуудсаар харуулна.
    python scripts/serve.py                 # http://127.0.0.1:8000
    python scripts/serve.py --no-predictor  # зөвхөн хуудас (таамаглалыг өөр процесс гаргана)
    python scripts/serve.py --candles-csv X.csv --calendar-json Y.json --now "2025-04-04 11:00"  # офлайн демо

API:
    GET /api/latest   -> data/live/latest.json (лаа, 5 хувилбарын зам, магадлал, ойрын мэдээ)
    GET /api/history  -> таамаглалын бүртгэл, бодит үр дүнтэй нь
    GET /api/health   -> сүүлийн шинэчлэлт, алдаа
"""
import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import predict_live as pl  # noqa: E402

WEB = ROOT / "web"
state = {"last_run": None, "last_error": None, "runs": 0}
app = FastAPI(title="EUR/USD хувилбарын таамаглал")
app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


@app.get("/api/latest")
def latest():
    f = pl.LIVE / "latest.json"
    if not f.exists():
        raise HTTPException(503, "Таамаглал хараахан гараагүй байна. Хэдэн секунд хүлээгээд дахин ачаална уу.")
    return FileResponse(f, media_type="application/json", headers={"Cache-Control": "no-store"})


@app.get("/api/history")
def history(n: int = 200):
    return JSONResponse(pl.history_json(n), headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return state


def predictor_loop(args):
    while True:
        try:
            pl.run_once(args)
            state["last_run"] = str(pd.Timestamp.now(tz="UTC"))
            state["last_error"] = None
            state["runs"] += 1
        except BaseException as e:  # noqa: BLE001  (sys.exit-ийг ч барина)
            state["last_error"] = str(e)
            print(f"[алдаа] {e}")
        if args.now:          # офлайн демо: нэг удаа
            return
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor("h") + pd.Timedelta(hours=1, minutes=2)
        time.sleep(max(30, (nxt - now).total_seconds()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-predictor", action="store_true")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--candles-csv")
    ap.add_argument("--calendar-json")
    ap.add_argument("--now")
    args = ap.parse_args()
    if not args.no_predictor:
        threading.Thread(target=predictor_loop, args=(args,), daemon=True).start()
    print(f"Чарт: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
