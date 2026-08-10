# app/routers.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException, Query
from app.services import (
    hitung_analisis_saham, hitung_momentum_gorengan, cek_kondisi_market,
    ambil_riwayat_batch, hitung_sinyal_fast_intraday, hitung_sinyal_bsjp,
    pre_filter_oversold_swing, pre_filter_momentum
)
from app.backtest import (
    backtest_swing_dividen, backtest_gorengan_momentum,
    backtest_watchlist_swing, backtest_watchlist_gorengan,
    backtest_fast_intraday, backtest_watchlist_fast_intraday,
    backtest_bsjp, backtest_watchlist_bsjp
)
from app.config import (
    INDEX_BLUECHIP_UTAMA, WATCHLIST_GORENGAN, WATCHLIST_FAST_INTRADAY,
    INTERVAL_FAST_INTRADAY, PERIODE_DATA_FAST_INTRADAY,
    VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY, JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY,
    WATCHLIST_BSJP, BSJP_PERIODE_DATA
)
from app.daftar_saham_bei import SEMUA_SAHAM_BEI

router = APIRouter(prefix="/v1")

# ThreadPoolExecutor dipakai untuk paralelisasi network I/O (yfinance). Angka worker
# dijaga moderat supaya tidak kena rate-limit Yahoo Finance.
MAX_WORKERS_SCREENER = 5

# Batas ukuran batch untuk endpoint "screener semua saham IDX". Vercel serverless
# punya batas waktu eksekusi (10-60 detik tergantung plan) - scan >900 saham IDX
# TIDAK MUAT dalam satu request. Karena itu endpoint ini dipaginasi: panggil
# berkali-kali dengan offset berbeda (atau jadwalkan lewat n8n/cron) untuk cover
# seluruh universe secara bertahap.
BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN = 60


@router.get("/analisis/swing/{ticker}")
async def analisis_swing_saham(ticker: str):
    """Analisis lengkap satu saham (fundamental + teknikal) untuk strategi swing-dividen."""
    data = hitung_analisis_saham(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


@router.get("/analisis/gorengan/{ticker}")
async def analisis_gorengan_saham(ticker: str):
    """Analisis momentum satu saham untuk strategi day-trading ADX."""
    data = hitung_momentum_gorengan(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


@router.get("/analisis/fast-intraday/{ticker}")
async def analisis_fast_intraday_saham(ticker: str):
    """
    Sinyal fast-intraday (15 menit) satu saham. BUKAN scalping asli — lihat field
    'peringatan_bukan_scalping_asli' di response untuk keterbatasannya.
    """
    data = hitung_sinyal_fast_intraday(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data 15-menit untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


@router.get("/market/status")
async def status_market():
    """Cek kondisi tren IHSG saat ini sebagai filter makro sebelum sinyal individual dipakai."""
    return {"status": "success", "data": cek_kondisi_market()}


def _jalankan_screener_swing(daftar_ticker: list):
    """
    Screener swing generik TEROPTIMASI: fetch riwayat harga SEMUA ticker sekaligus
    lewat batch download, lalu PRE-FILTER teknikal secara lokal (CPU-only, tanpa HTTP).
    Hanya saham yang lolos pre-filter yang diproses lengkap (termasuk fetch .info dari
    Yahoo Finance) — menghemat puluhan HTTP request yang sebelumnya terbuang untuk
    saham yang tidak lolos filter oversold.
    """
    kondisi_market = cek_kondisi_market()
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period="1y", interval="1d")

    # TAHAP 1: Pre-filter teknikal (CPU-only, tanpa HTTP) — eliminasi saham yang
    # jelas tidak lolos filter oversold sebelum membuang waktu download .info.
    ticker_lolos_prefilter = []
    saham_gagal = []
    for t in tickers_jk:
        df = data_riwayat.get(t)
        if df is None or df.empty or len(df) < 200:
            saham_gagal.append(t.replace(".JK", ""))
            continue
        if pre_filter_oversold_swing(df):
            ticker_lolos_prefilter.append(t)

    # TAHAP 2: Analisis lengkap (termasuk .info) HANYA untuk yang lolos pre-filter.
    saham_lolos = []

    def proses(ticker_jk):
        symbol = ticker_jk.replace(".JK", "")
        df_riwayat = data_riwayat.get(ticker_jk)
        try:
            return symbol, hitung_analisis_saham(symbol, kondisi_market=kondisi_market, df_riwayat=df_riwayat)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCREENER) as executor:
        futures = [executor.submit(proses, t) for t in ticker_lolos_prefilter]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                saham_gagal.append(symbol)
                continue
            teknikal = data.get("teknikal", {})
            if teknikal.get("oversold_swing_aktif"):
                saham_lolos.append({
                    "saham": data.get("saham"),
                    "harga_saat_ini": data.get("harga_saat_ini"),
                    "status_tren": teknikal.get("status_tren"),
                    "konfirmasi_oversold_swing": teknikal.get("konfirmasi_oversold_swing"),
                    "arus_bandar_cmf": teknikal.get("arus_bandar_cmf"),
                    "status_dividen": data.get("fundamental", {}).get("status_dividen"),
                    "zona_average_down": data.get("zona_average_down"),
                    "guardrail_fundamental": data.get("guardrail_fundamental"),
                    "manajemen_risiko": data.get("manajemen_risiko"),
                    "rekomendasi": data.get("rekomendasi_akhir")
                })

    return kondisi_market, saham_lolos, saham_gagal


def _jalankan_screener_gorengan(daftar_ticker: list):
    """Screener gorengan generik TEROPTIMASI dengan batch fetch + pre-filter teknikal."""
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period="60d", interval="1h")

    # TAHAP 1: Pre-filter teknikal (CPU-only, tanpa HTTP)
    ticker_lolos_prefilter = []
    saham_gagal = []
    for t in tickers_jk:
        df = data_riwayat.get(t)
        if df is None or df.empty or len(df) < 40:
            saham_gagal.append(t.replace(".JK", ""))
            continue
        if pre_filter_momentum(df, ema_span_pendek=5, ema_span_panjang=10,
                                volume_multiplier=2.5, volume_lookback=35):
            ticker_lolos_prefilter.append(t)

    # TAHAP 2: Analisis lengkap HANYA untuk yang lolos pre-filter.
    saham_lolos = []

    def proses(ticker_jk):
        symbol = ticker_jk.replace(".JK", "")
        df_riwayat = data_riwayat.get(ticker_jk)
        try:
            return symbol, hitung_momentum_gorengan(symbol, df_riwayat=df_riwayat)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCREENER) as executor:
        futures = [executor.submit(proses, t) for t in ticker_lolos_prefilter]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                saham_gagal.append(symbol)
                continue
            if "LOLOS" in data.get("status_filter", ""):
                saham_lolos.append({
                    "saham": data.get("saham"),
                    "status": data.get("status_filter"),
                    "rsi_momentum": data.get("indikator", {}).get("rsi_momentum"),
                    "adx_power": data.get("indikator", {}).get("adx_power"),
                    "kualitas_tren_adx": data.get("indikator", {}).get("kualitas_tren_adx"),
                    "arus_bandar_cmf": data.get("indikator", {}).get("arus_bandar_cmf"),
                    "rekomendasi_entry_daytrading": data.get("rekomendasi_entry_daytrading"),
                    "bracket_order_growin": data.get("bracket_order_growin"),
                    "manajemen_risiko": data.get("manajemen_risiko"),
                    "rekomendasi": data.get("rekomendasi_aksi")
                })

    return saham_lolos, saham_gagal


@router.get("/screener/swing-dividen")
async def run_screener_swing_dividen():
    kondisi_market, saham_lolos, saham_gagal = _jalankan_screener_swing(INDEX_BLUECHIP_UTAMA)
    return {
        "status": "success",
        "kondisi_market": kondisi_market,
        "data": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/gorengan-momentum")
async def run_screener_gorengan_momentum():
    saham_lolos, saham_gagal = _jalankan_screener_gorengan(WATCHLIST_GORENGAN)
    return {
        "status": "success",
        "radar_saham_gorengan_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/swing-dividen/semua-saham")
async def run_screener_swing_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """
    Screener swing-dividen atas SELURUH daftar saham tercatat BEI (bukan cuma
    watchlist bluechip). DIPAGINASI karena scan seluruh saham IDX tidak muat dalam
    1 request serverless — panggil berulang dengan offset berbeda untuk cover semua.
    Contoh: offset=0&limit=50, lalu offset=50&limit=50, dst.
    """
    total_saham = len(SEMUA_SAHAM_BEI)
    slice_ticker = SEMUA_SAHAM_BEI[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di daftar BEI: {total_saham}.")

    kondisi_market, saham_lolos, saham_gagal = _jalankan_screener_swing(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "kondisi_market": kondisi_market,
        "total_saham_di_daftar_bei": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "data": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/gorengan-momentum/semua-saham")
async def run_screener_gorengan_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """Sama seperti swing/semua-saham tapi untuk strategi momentum gorengan. Juga dipaginasi."""
    total_saham = len(SEMUA_SAHAM_BEI)
    slice_ticker = SEMUA_SAHAM_BEI[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di daftar BEI: {total_saham}.")

    saham_lolos, saham_gagal = _jalankan_screener_gorengan(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "total_saham_di_daftar_bei": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "radar_saham_gorengan_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/backtest/swing/{ticker}")
async def backtest_swing(ticker: str, tahun: int = 2):
    """Backtest strategi swing-oversold pada data historis harian 1 saham."""
    hasil = backtest_swing_dividen(ticker, tahun=tahun)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/gorengan/{ticker}")
async def backtest_gorengan(ticker: str):
    """Backtest strategi momentum gorengan pada data historis 60 hari interval 1 jam."""
    hasil = backtest_gorengan_momentum(ticker)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/swing/watchlist/gabungan")
async def backtest_swing_gabungan(tahun: int = 2):
    """Backtest strategi swing-oversold di SELURUH watchlist bluechip sekaligus, digabungkan (lebih valid secara statistik)."""
    hasil = backtest_watchlist_swing(tahun=tahun)
    return {"status": "success", "data": hasil}


@router.get("/backtest/gorengan/watchlist/gabungan")
async def backtest_gorengan_gabungan():
    """Backtest strategi momentum gorengan di SELURUH watchlist gorengan sekaligus, digabungkan."""
    hasil = backtest_watchlist_gorengan()
    return {"status": "success", "data": hasil}


def _jalankan_screener_fast_intraday(daftar_ticker: list):
    """Screener fast-intraday generik TEROPTIMASI dengan batch fetch + pre-filter teknikal."""
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period=PERIODE_DATA_FAST_INTRADAY, interval=INTERVAL_FAST_INTRADAY)

    # TAHAP 1: Pre-filter teknikal (CPU-only, tanpa HTTP)
    ticker_lolos_prefilter = []
    saham_gagal = []
    for t in tickers_jk:
        df = data_riwayat.get(t)
        if df is None or df.empty or len(df) < 40:
            saham_gagal.append(t.replace(".JK", ""))
            continue
        if pre_filter_momentum(df, ema_span_pendek=5, ema_span_panjang=13,
                                volume_multiplier=VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY,
                                volume_lookback=JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY):
            ticker_lolos_prefilter.append(t)

    # TAHAP 2: Analisis lengkap HANYA untuk yang lolos pre-filter.
    saham_lolos = []

    def proses(ticker_jk):
        symbol = ticker_jk.replace(".JK", "")
        df_riwayat = data_riwayat.get(ticker_jk)
        try:
            return symbol, hitung_sinyal_fast_intraday(symbol, df_riwayat=df_riwayat)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS_SCREENER) as executor:
        futures = [executor.submit(proses, t) for t in ticker_lolos_prefilter]
        for future in as_completed(futures):
            symbol, data = future.result()
            if not data:
                saham_gagal.append(symbol)
                continue
            if "LOLOS" in data.get("status_filter", ""):
                saham_lolos.append({
                    "saham": data.get("saham"),
                    "status": data.get("status_filter"),
                    "rsi_momentum": data.get("indikator", {}).get("rsi_momentum"),
                    "adx_power": data.get("indikator", {}).get("adx_power"),
                    "kualitas_tren_adx": data.get("indikator", {}).get("kualitas_tren_adx"),
                    "arus_bandar_cmf": data.get("indikator", {}).get("arus_bandar_cmf"),
                    "rekomendasi_entry_daytrading": data.get("rekomendasi_entry_daytrading"),
                    "bracket_order_growin": data.get("bracket_order_growin"),
                    "rekomendasi": data.get("rekomendasi_aksi")
                })

    return saham_lolos, saham_gagal


@router.get("/screener/fast-intraday")
async def run_screener_fast_intraday():
    """
    Screener fast-intraday (15 menit) atas watchlist saham LIKUID (bukan gorengan).
    BUKAN scalping asli — lihat 'peringatan_bukan_scalping_asli' di tiap hasil analisis.
    """
    saham_lolos, saham_gagal = _jalankan_screener_fast_intraday(WATCHLIST_FAST_INTRADAY)
    return {
        "status": "success",
        "peringatan": "Sinyal 15-menit, BUKAN scalping real-time. Data Yahoo Finance delay ~15-20 menit untuk saham IDX — verifikasi harga di broker sebelum eksekusi.",
        "radar_saham_fast_intraday_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/fast-intraday/semua-saham")
async def run_screener_fast_intraday_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """Sama seperti screener fast-intraday, tapi atas SELURUH daftar saham BEI. Dipaginasi."""
    total_saham = len(SEMUA_SAHAM_BEI)
    slice_ticker = SEMUA_SAHAM_BEI[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di daftar BEI: {total_saham}.")

    saham_lolos, saham_gagal = _jalankan_screener_fast_intraday(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "total_saham_di_daftar_bei": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "radar_saham_fast_intraday_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/backtest/fast-intraday/{ticker}")
async def backtest_fast_intraday_endpoint(ticker: str):
    """Backtest strategi fast-intraday pada data historis 60 hari interval 15 menit."""
    hasil = backtest_fast_intraday(ticker)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis 15-menit untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/fast-intraday/watchlist/gabungan")
async def backtest_fast_intraday_gabungan():
    """Backtest strategi fast-intraday di SELURUH watchlist likuid sekaligus, digabungkan."""
    hasil = backtest_watchlist_fast_intraday()
    return {"status": "success", "data": hasil}


# =========================================================================
# STRATEGI 4: BSJP (Beli Sore Jual Pagi)
# =========================================================================

@router.get("/analisis/bsjp/{ticker}")
async def analisis_bsjp_saham(ticker: str):
    """
    Sinyal BSJP (Beli Sore Jual Pagi) satu saham: beli di sesi penutupan hari ini,
    jual di pembukaan besok. Berbasis data harian + statistik gap historis.
    """
    data = hitung_sinyal_bsjp(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": data}


def _jalankan_screener_bsjp(daftar_ticker: list):
    """
    Screener BSJP: batch fetch data harian seluruh ticker sekaligus, lalu evaluasi
    sinyal secara lokal (CPU-only, tanpa HTTP .info — BSJP murni teknikal).
    """
    tickers_jk = [t if t.endswith(".JK") else f"{t.upper()}.JK" for t in daftar_ticker]
    data_riwayat = ambil_riwayat_batch(tickers_jk, period=BSJP_PERIODE_DATA, interval="1d")

    saham_lolos = []
    saham_gagal = []
    for t in tickers_jk:
        df = data_riwayat.get(t)
        if df is None or df.empty or len(df) < 60:
            saham_gagal.append(t.replace(".JK", ""))
            continue
        try:
            data = hitung_sinyal_bsjp(t.replace(".JK", ""), df_riwayat=df)
        except Exception:
            data = None
        if not data:
            saham_gagal.append(t.replace(".JK", ""))
            continue
        if "LOLOS" in data.get("status_filter", ""):
            saham_lolos.append({
                "saham": data.get("saham"),
                "harga_saat_ini": data.get("harga_saat_ini"),
                "status": data.get("status_filter"),
                "detail_sinyal": data.get("detail_sinyal_hari_ini"),
                "rekomendasi_bsjp": data.get("rekomendasi_bsjp"),
                "statistik_gap_historis": data.get("statistik_gap_historis")
            })
    return saham_lolos, saham_gagal


@router.get("/screener/bsjp")
async def run_screener_bsjp():
    """
    Screener BSJP atas watchlist (BBRI dkk). Jalankan ~15.30-15.45 WIB untuk
    keputusan beli sebelum pasar tutup; exit besok pagi dilakukan manual.
    """
    saham_lolos, saham_gagal = _jalankan_screener_bsjp(WATCHLIST_BSJP)
    return {
        "status": "success",
        "peringatan": (
            "Sinyal BSJP berbasis data harian Yahoo (delay ~15-20 menit). Jalankan paling cepat "
            "~15.30-15.45 WIB dan verifikasi harga real-time di broker sebelum eksekusi. "
            "Exit besok pagi (09.00-09.30 WIB) manual."
        ),
        "radar_bsjp_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/screener/bsjp/semua-saham")
async def run_screener_bsjp_semua_saham(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=BATAS_LIMIT_MAKSIMAL_PER_PANGGILAN)):
    """Screener BSJP atas SELURUH daftar saham BEI. Dipaginasi seperti screener lain."""
    total_saham = len(SEMUA_SAHAM_BEI)
    slice_ticker = SEMUA_SAHAM_BEI[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di daftar BEI: {total_saham}.")

    saham_lolos, saham_gagal = _jalankan_screener_bsjp(slice_ticker)
    offset_berikutnya = offset + limit if (offset + limit) < total_saham else None

    return {
        "status": "success",
        "total_saham_di_daftar_bei": total_saham,
        "diproses_offset": offset,
        "diproses_limit": limit,
        "offset_berikutnya": offset_berikutnya,
        "radar_bsjp_aktif": saham_lolos,
        "saham_gagal_dianalisis": saham_gagal
    }


@router.get("/backtest/bsjp/{ticker}")
async def backtest_bsjp_endpoint(ticker: str, tahun: int = Query(2, ge=1, le=5)):
    """
    Backtest BSJP: beli di close hari sinyal -> jual di open hari berikutnya,
    potong fee. Data harian -> sampel lebih robust dari backtest intraday.
    """
    hasil = backtest_bsjp(ticker, periode_tahun=tahun)
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/bsjp/watchlist/gabungan")
async def backtest_bsjp_gabungan(tahun: int = Query(2, ge=1, le=5)):
    """Backtest BSJP di SELURUH watchlist BSJP sekaligus, digabungkan (lebih valid secara statistik)."""
    hasil = backtest_watchlist_bsjp(periode_tahun=tahun)
    return {"status": "success", "data": hasil}
