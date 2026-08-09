# api/index.py
import sys
import os

# Jembatan pengenal agar Vercel bisa membaca folder app di luar folder api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- SOLUSI ERROR 500 VERCEL (filesystem read-only) ---
# Arahkan cache yfinance ke /tmp, satu-satunya folder yang bisa ditulis di Vercel
import yfinance as yf
yf.set_tz_cache_location("/tmp/py-yfinance-cache")

from fastapi import FastAPI
from app.routers import router

app = FastAPI(
    title="GROWIN Multi-Strategy Trading Engine",
    description="Sistem API Otomatis Penggabungan Strategi Swing Dividen & Day Trading ADX Forum v4.0",
    version="4.0"
)

app.include_router(router)

@app.get("/")
def check_status():
    return {"status": "active", "message": "Server v4.0 Online di Vercel! Siap kirim data ke n8n."}

# Wajib diekspos sebagai handler untuk Vercel Runtime
handler = app
