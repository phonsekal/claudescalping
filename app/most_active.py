# app/most_active.py
"""
Most Active / Top Value / Top Gainer / Top Loser + Rekomendasi Intraday.

Mengapa dihitung SENDIRI (bukan ambil dari IDX/Yahoo screener):
- API resmi idx.co.id (TradingSummary) diblokir Cloudflare (403) — riset Agu 2026.
- Screener Yahoo (region ID) rate-limited dari IP serverless.

Solusi: batch download SELURUH daftar BEI (941 ticker) sekali jalan
(~16-21 dtk lokal dengan chunk 250 + threads), lalu hitung metrik pasar dari
hari terakhir yang punya data. Data sama dengan yang dipakai semua strategi
lain di sistem ini — konsisten, tanpa sumber eksternal.

Keputusan desain (dibahas dengan user, Agu 2026):
- Most Active dipakai sebagai FILTER LIKUIDITAS saja. Arah keputusan tetap dari
  sinyal strategi (BPJS/BSJP/fast-intraday), bukan dari "saham lagi ramai".
  Uji empiris 62 saham likuid (2 tahun data harian): most active T+1 +0,29%
  (edge = likuiditas, bukan arah); top gainer justru cenderung REVERSAL
  (T+1 -0,07%) — karena itu gainer TIDAK dipakai sebagai sinyal beli.
- Data HARI SEBELUMNYA untuk screening/rekomendasi (lengkap, tanpa bias sesi,
  konsisten dengan backtest walk-forward). Data hari berjalan hanya untuk
  konfirmasi eksekusi (Yahoo delay ~15-20 menit untuk IDX).
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

from app.daftar_saham_bei import SEMUA_SAHAM_BEI
from app.services import rekomendasi_strategi_gabungan

# Konfigurasi batch download. period="5d" cukup untuk menghitung metrik pasar
# (harga terakhir, volume, nilai transaksi, perubahan hari & 5 hari).
BATCH_PERIODE = "5d"
BATCH_INTERVAL = "1d"
CHUNK_BESAR = 250

# Strategi intraday yang direkomendasikan dari filter most-active.
STRATEGI_INTRADAY = [
    ("bpjs", "BPJS (Beli Pagi Jual Sore)"),
    ("bsjp", "BSJP (Beli Sore Jual Pagi)"),
    ("range_pagi_sore", "Range Pagi-Sore (Jual Pagi, Beli Sore)"),
    ("fast_intraday", "Fast Intraday Alert (15 menit)"),
]

# Ambang likuiditas minimum untuk layak dianalisis intraday (nilai transaksi
# harian, Rupiah). Di bawah ini spread & slippage makan profit tipis intraday.
MIN_NILAI_TRANSAKSI_INTRADAY = 5_000_000_000  # Rp 5 miliar


def _nama_bisa(t: str) -> str:
    """Hapus suffix .JK untuk tampilan yang rapi."""
    return t[:-3] if t.endswith(".JK") else t


def _ekstrak_metrik(t: str, df: pd.DataFrame) -> dict:
    """
    Metrik pasar satu saham dari DataFrame harian.

    Penting (keputusan desain yang dibahas dengan user): SCREENING memakai data
    HARI TRADING LENGKAP TERAKHIR, bukan sesi hari ini yang belum selesai (volume
    parsial akan merusak peringkat most-active). Kalau baris terakhir bertanggal
    hari ini WIB dan jam sekarang < 16:00 (sesi masih berjalan/jeda), pakai baris
    trading lengkap sebelumnya.
    """
    d = df.dropna(how="all")
    if d.empty:
        return None

    # Hanya baris dengan harga & volume yang benar-benar terisi (>0).
    mask = d["Close"].notna() & d["Volume"].notna() & (d["Volume"] > 0)
    valid = d[mask]
    if valid.empty:
        return None

    baris = valid.iloc[-1]
    now_jkt = datetime.now(ZoneInfo("Asia/Jakarta"))
    # Sesi berjalan (data hari ini belum final) -> geser ke hari trading lengkap.
    if (hasattr(baris.name, "date") and baris.name.date() == now_jkt.date()
            and now_jkt.hour < 16):
        prev = valid.iloc[:-1]
        if prev.empty:
            return None
        baris = prev.iloc[-1]

    harga = float(baris["Close"])
    volume = float(baris["Volume"])
    nilai = harga * volume
    tanggal = str(baris.name.date()) if hasattr(baris.name, "date") else str(baris.name)

    idx = valid.index.get_loc(baris.name)
    pct_hari = None
    if idx >= 1:
        prev_close = float(valid.iloc[idx - 1]["Close"])
        if prev_close > 0:
            pct_hari = round((harga / prev_close - 1) * 100, 2)

    pct_5d = None
    if len(valid) >= 2:
        base = float(valid.iloc[0]["Close"])
        if base > 0:
            pct_5d = round((harga / base - 1) * 100, 2)

    avg_volume = float(valid.iloc[:idx + 1]["Volume"].mean())

    return {
        "ticker": _nama_bisa(t),
        "harga": round(harga),
        "volume": int(volume),
        "volume_rata_rata_5d": int(avg_volume),
        "nilai_transaksi": int(nilai),
        "pct_change_hari": pct_hari,
        "pct_change_5d": pct_5d,
        "tanggal": tanggal,
    }


def ambil_metrik_pasar() -> dict:
    """
    Batch download seluruh daftar BEI (5d, harian) dan hitung metrik pasar.

    Return: {"metrik": {ticker: {...}}, "tanggal_data": ..., "jumlah_berhasil": n}
    Retry 1x per chunk untuk menyerap rate-limit Yahoo sesaat.
    """
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in SEMUA_SAHAM_BEI]
    metrik = {}
    chunk_gagal = []

    for i in range(0, len(tickers_jk), CHUNK_BESAR):
        chunk = tickers_jk[i:i + CHUNK_BESAR]
        for percobaan in range(2):
            try:
                data = yf.download(
                    chunk, period=BATCH_PERIODE, interval=BATCH_INTERVAL,
                    group_by="ticker", auto_adjust=False, threads=True, progress=False
                )
                break
            except Exception:
                if percobaan == 0:
                    time.sleep(2.0)
                else:
                    data = None
        if data is None:
            # Seluruh chunk gagal (rate-limit Yahoo) — catat supaya output bisa
            # memberi tahu user bahwa ranking tidak lengkap.
            chunk_gagal.append(len(chunk))
            continue
        for t in chunk:
            try:
                if t not in data or data[t] is None:
                    continue
                m = _ekstrak_metrik(t, data[t])
                if m:
                    metrik[t] = m
            except Exception:
                continue
        # Spacing antar chunk: sebarkan burst request supaya Yahoo tidak 429.
        time.sleep(0.5)

    # Tanggal data = tanggal yang PALING SERING muncul (bukan tanggal saham
    # pertama — saham tipis bisa punya tanggal lama dan menyesatkan header).
    tanggal = None
    if metrik:
        from collections import Counter
        counter = Counter(m["tanggal"] for m in metrik.values())
        tanggal = counter.most_common(1)[0][0]

    return {
        "metrik": metrik,
        "tanggal_data": tanggal,
        "jumlah_berhasil": len(metrik),
        "chunk_gagal": chunk_gagal,
    }


def top_pasar(jenis: str = "active", limit: int = 10) -> dict:
    """
    Dashboard pasar: top saham per kategori metrik.

    jenis: active (volume terbesar) | value (nilai transaksi terbesar) |
           gainer (perubahan % tertinggi) | loser (perubahan % terendah)
    """
    hasil = ambil_metrik_pasar()
    metrik = hasil["metrik"]
    daftar = list(metrik.values())

    def _kunci_aktif(m):
        return m["volume"]

    def _kunci_nilai(m):
        return m["nilai_transaksi"]

    def _kunci_gainer(m):
        return m["pct_change_hari"] if m["pct_change_hari"] is not None else -999

    def _kunci_loser(m):
        return m["pct_change_hari"] if m["pct_change_hari"] is not None else 999

    pilihan = {
        "active": (_kunci_aktif, True, "Most Active (volume terbesar)"),
        "value": (_kunci_nilai, True, "Top Value (nilai transaksi terbesar)"),
        "gainer": (_kunci_gainer, True, "Top Gainer (kenaikan % tertinggi)"),
        "loser": (_kunci_loser, False, "Top Loser (penurunan % terbesar)"),
    }
    if jenis not in pilihan:
        raise ValueError(f"jenis tidak dikenal: {jenis}. Gunakan active|value|gainer|loser")

    kunci, reverse, label = pilihan[jenis]
    daftar.sort(key=kunci, reverse=reverse)
    # Gainer/loser butuh pct_change yang benar-benar ada.
    if jenis in ("gainer", "loser"):
        daftar = [m for m in daftar if m["pct_change_hari"] is not None]
    daftar = daftar[:limit]

    peringatan = None
    if hasil.get("chunk_gagal"):
        peringatan = (
            f"{sum(hasil['chunk_gagal'])} saham gagal diunduh (rate-limit data Yahoo) "
            "— sebagian saham tidak ikut dalam ranking ini."
        )

    return {
        "jenis": jenis,
        "label": label,
        "tanggal_data": hasil["tanggal_data"],
        "jumlah_saham_dianalisis": hasil["jumlah_berhasil"],
        "peringatan": peringatan,
        "list": daftar,
    }


def rekomendasi_intraday_likuid(n: int = 8, jenis: str = "active", max_workers: int = 3) -> dict:
    """
    Rekomendasi saham UNTUK STRATEGI INTRADAY, berbasis filter most-active.

    Alur:
    1. Batch download seluruh BEI -> ambil top-n saham paling likuid (most active
       atau top value). Ini FILTER LIKUIDITAS — bukan sinyal arah.
    2. Untuk tiap kandidat: jalankan rekomendasi_strategi_gabungan (skor
       kecocokan 6 strategi dari data yang ditarik sekali).
    3. Ranking per strategi intraday (BPJS, BSJP, range-pagi-sore, fast-intraday):
       saham skor tertinggi = paling cocok untuk strategi itu SEKARANG.

    backtest sengaja TIDAK dijalankan (beban request besar di serverless) —
    skor + sinyal aktif sudah cukup untuk rekomendasi; validasi historis bisa
    dilakukan per saham lewat /v1/backtest/validasi-strategi/{ticker}.
    """
    pasar = ambil_metrik_pasar()
    metrik = pasar["metrik"]

    # Filter likuiditas minimum: nilai transaksi >= Rp 5 miliar (layak intraday).
    kandidat = [m for m in metrik.values() if m["nilai_transaksi"] >= MIN_NILAI_TRANSAKSI_INTRADAY]
    if jenis == "value":
        kandidat.sort(key=lambda m: -m["nilai_transaksi"])
    else:
        kandidat.sort(key=lambda m: -m["volume"])
    kandidat = kandidat[:n]
    ticker_list = [m["ticker"] for m in kandidat]

    if not ticker_list:
        return {
            "filter": {"jenis": jenis, "label": "Most Active" if jenis != "value" else "Top Value"},
            "jumlah_saham_diproses": 0,
            "jumlah_saham_berhasil": 0,
            "saham_gagal": [],
            "kandidat": [],
            "leaderboard_per_strategi_intraday": [],
            "peringatan": "Tidak ada saham dengan nilai transaksi >= Rp5 miliar.",
        }

    def _proses(t):
        # Retry pendek dengan backoff untuk menyerap rate-limit Yahoo sesaat.
        for percobaan in range(3):
            try:
                return t, rekomendasi_strategi_gabungan(t)
            except Exception:
                if percobaan < 2:
                    time.sleep(1.5 * (percobaan + 1))
        return t, None

    def _jalankan_batch(daftar, worker):
        ok = {}
        gag = []
        with ThreadPoolExecutor(max_workers=worker) as ex:
            futures = []
            for t in daftar:
                futures.append(ex.submit(_proses, t))
                time.sleep(0.4)  # spacing antar submit, anti burst 429
            for fut in as_completed(futures):
                t, r = fut.result()
                if r:
                    ok[t] = r
                else:
                    gag.append(t)
        return ok, gag

    gabungan = {}
    ok1, gagal = _jalankan_batch(ticker_list, max_workers)
    gabungan.update(ok1)
    if gagal and len(gabungan) < len(ticker_list):
        time.sleep(2.0)
        ok2, gagal = _jalankan_batch(gagal, max(1, min(max_workers, 2)))
        gabungan.update(ok2)

    # Leaderboard per strategi intraday dari skor gabungan.
    skor_by = {kode: [] for kode, _ in STRATEGI_INTRADAY}
    for t, g in gabungan.items():
        for p in g.get("peringkat_strategi", []):
            if p["kode"] in skor_by:
                skor_by[p["kode"]].append({
                    "ticker": _nama_bisa(t),
                    "skor": p["skor"],
                    "kecocokan": p["kecocokan"],
                    "sinyal_aktif": p["sinyal_aktif"],
                    "harga": g.get("harga_saat_ini"),
                })

    leaderboard = []
    for kode, nama in STRATEGI_INTRADAY:
        lst = sorted(skor_by[kode], key=lambda x: -x["skor"])
        leaderboard.append({"kode": kode, "nama": nama, "terbaik": lst[:3]})

    if not gabungan:
        peringatan = (
            "Semua kandidat gagal dianalisis — kemungkinan besar rate-limit data Yahoo "
            "pada IP server. Coba lagi 1-2 menit kemudian, atau kurangi n (contoh: n=5)."
        )
    else:
        peringatan = None

    if pasar.get("chunk_gagal") and not peringatan:
        peringatan = (
            f"{sum(pasar['chunk_gagal'])} saham gagal diunduh saat batch pasar "
            "(rate-limit Yahoo) — kandidat mungkin tidak mencakup semua saham likuid."
        )

    return {
        "filter": {
            "jenis": jenis,
            "label": "Most Active (volume terbesar)" if jenis != "value" else "Top Value (nilai transaksi)",
            "deskripsi": "Most Active dipakai sebagai FILTER LIKUIDITAS saja — arah keputusan tetap dari sinyal strategi.",
        },
        "tanggal_data": pasar["tanggal_data"],
        "jumlah_saham_dianalisis": pasar["jumlah_berhasil"],
        "jumlah_saham_diproses": len(ticker_list),
        "jumlah_saham_berhasil": len(gabungan),
        "saham_gagal": [_nama_bisa(t) for t in gagal],
        "kandidat": [
            {
                "ticker": m["ticker"],
                "harga": m["harga"],
                "volume": m["volume"],
                "nilai_transaksi_miliar": round(m["nilai_transaksi"] / 1e9, 1),
                "pct_change_hari": m["pct_change_hari"],
            }
            for m in kandidat
        ],
        "leaderboard_per_strategi_intraday": leaderboard,
        "peringatan": peringatan,
    }
