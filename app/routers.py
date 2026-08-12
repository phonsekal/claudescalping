# app/routers.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import APIRouter, HTTPException, Query
from app.services import (
    hitung_analisis_saham, hitung_momentum_gorengan, cek_kondisi_market,
    ambil_riwayat_batch, hitung_sinyal_fast_intraday, hitung_sinyal_bsjp,
    hitung_analisis_range_pagi_sore, hitung_analisis_bpjs,
    rekomendasi_strategi_gabungan,
    pre_filter_oversold_swing, pre_filter_momentum
)
from app.backtest import (
    backtest_swing_dividen, backtest_gorengan_momentum,
    backtest_watchlist_swing, backtest_watchlist_gorengan,
    backtest_fast_intraday, backtest_watchlist_fast_intraday,
    backtest_bsjp, backtest_watchlist_bsjp,
    backtest_range_pagi_sore, backtest_bpjs
)
from app.validasi_strategi import validasi_kecocokan_satu_saham
from app.config import (
    INDEX_BLUECHIP_UTAMA, WATCHLIST_GORENGAN, WATCHLIST_FAST_INTRADAY,
    INTERVAL_FAST_INTRADAY, PERIODE_DATA_FAST_INTRADAY,
    VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY, JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY,
    WATCHLIST_BSJP, BSJP_PERIODE_DATA
)
from app.daftar_saham_bei import SEMUA_SAHAM_BEI
from app.most_active import (
    top_pasar, rekomendasi_intraday_likuid, digest_pagi, konteks_pasar_saham,
    screener_range_pagi_sore,
)
from app.riwayat import ambil_riwayat_digest

router = APIRouter(prefix="/v1")


def _tambah_konteks_pasar(data: dict, ticker: str) -> dict:
    """
    Lampirkan badge konteks pasar (Most Active / Top Value / Top Gainer / Top
    Loser + nilai transaksi + kelayakan intraday) ke hasil analisis saham.

    Best-effort & cepat: memakai cache batch pasar (KV Upstash); kalau cache
    belum ada, fallback menghitung metrik SATU saham saja. Kegagalan apa pun
    diabaikan — analisis utama tetap berjalan normal tanpa badge.
    """
    try:
        k = konteks_pasar_saham(ticker)
        if k:
            data["konteks_pasar"] = k
    except Exception:
        pass
    return data

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
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


@router.get("/analisis/gorengan/{ticker}")
async def analisis_gorengan_saham(ticker: str):
    """Analisis momentum satu saham untuk strategi day-trading ADX."""
    data = hitung_momentum_gorengan(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


@router.get("/analisis/fast-intraday/{ticker}")
async def analisis_fast_intraday_saham(ticker: str):
    """
    Sinyal fast-intraday (15 menit) satu saham. BUKAN scalping asli — lihat field
    'peringatan_bukan_scalping_asli' di response untuk keterbatasannya.
    """
    data = hitung_sinyal_fast_intraday(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data 15-menit untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


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
async def analisis_bsjp_saham(
    ticker: str,
    gain_min: float = Query(None, ge=0, le=30, description="Kenaikan min % (None = default 3)"),
    close_pos: float = Query(None, ge=0, le=1, description="Posisi close min (None = default 0.7)"),
    vol_mult: float = Query(None, ge=0, le=50, description="Volume min x rata-rata (None = default 2)"),
    value_min: float = Query(None, ge=0, description="Nilai transaksi min Rupiah (None = default 5 miliar)"),
    rsi_max: float = Query(None, ge=0, le=100, description="RSI maks (None = default 85; 100 = nonaktif)"),
    adx_min: float = Query(None, ge=0, le=100, description="ADX min (None = default 20; 0 = nonaktif)"),
    tp_persen: float = Query(None, ge=0, le=50, description="Target TP % (None = default 3)")
):
    """
    Sinyal BSJP (Beli Sore Jual Pagi) satu saham: beli di sesi penutupan hari ini,
    jual di pembukaan besok. Berbasis data harian + statistik gap historis.

    Kriteria bisa dilonggarkan via query params — contoh varian longgar BBRI:
    ?gain_min=2&close_pos=0.6&vol_mult=1.5&rsi_max=100&adx_min=0&tp_persen=3
    """
    data = hitung_sinyal_bsjp(
        ticker, gain_min_persen=gain_min, close_posisi_min=close_pos,
        volume_multiplier=vol_mult, value_min_rupiah=value_min,
        rsi_maks=rsi_max, adx_min=adx_min, target_persen=tp_persen
    )
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


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
async def backtest_bsjp_endpoint(
    ticker: str,
    tahun: int = Query(2, ge=1, le=5),
    gain_min: float = Query(None, ge=0, le=30),
    close_pos: float = Query(None, ge=0, le=1),
    vol_mult: float = Query(None, ge=0, le=50),
    value_min: float = Query(None, ge=0),
    rsi_max: float = Query(None, ge=0, le=100),
    adx_min: float = Query(None, ge=0, le=100),
    tp_persen: float = Query(None, ge=0, le=50, description="Target TP % untuk statistik ketercapaian")
):
    """
    Backtest BSJP: beli di close hari sinyal -> jual di open hari berikutnya,
    potong fee. Data harian -> sampel lebih robust dari backtest intraday.

    Kriteria bisa dilonggarkan via query params — contoh varian longgar BBRI:
    ?gain_min=2&close_pos=0.6&vol_mult=1.5&rsi_max=100&adx_min=0&tp_persen=3
    """
    hasil = backtest_bsjp(
        ticker, periode_tahun=tahun,
        gain_min_persen=gain_min, close_posisi_min=close_pos,
        volume_multiplier=vol_mult, value_min_rupiah=value_min,
        rsi_maks=rsi_max, adx_min=adx_min, target_persen=tp_persen
    )
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data historis untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


@router.get("/backtest/bsjp/watchlist/gabungan")
async def backtest_bsjp_gabungan(tahun: int = Query(2, ge=1, le=5)):
    """Backtest BSJP di SELURUH watchlist BSJP sekaligus, digabungkan (lebih valid secara statistik)."""
    hasil = backtest_watchlist_bsjp(periode_tahun=tahun)
    return {"status": "success", "data": hasil}


# =========================================================================
# STRATEGI 5: RANGE PAGI-SORE (JUAL PAGI, BELI SORE)
# =========================================================================

@router.get("/analisis/range-pagi-sore/{ticker}")
async def analisis_range_pagi_sore_saham(
    ticker: str,
    window_hari: int = Query(None, ge=10, le=120, description="Jendela hari statistik (default 30)"),
    persentil_jual: float = Query(None, gt=0, lt=1, description="Persentil peak pagi -> level jual (default 0.3)"),
    persentil_beli: float = Query(None, gt=0, lt=1, description="Persentil trough sore -> level beli (default 0.5)")
):
    """
    Analisis pola Range Pagi-Sore untuk pemegang saham: rekomendasi harga pasang
    JUAL sebelum market buka (titik tinggi pagi) dan harga pasang BELI di sesi sore
    (titik rendah sore), berdasarkan statistik N hari sebelumnya + estimasi peluang
    terisi. Contoh: /v1/analisis/range-pagi-sore/BBRI
    """
    data = hitung_analisis_range_pagi_sore(
        ticker, window_hari=window_hari, persentil_jual=persentil_jual, persentil_beli=persentil_beli
    )
    if not data:
        raise HTTPException(status_code=404, detail=f"Data intraday untuk {ticker.upper()} tidak cukup untuk analisis pola.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


@router.get("/screener/range-pagi-sore")
async def run_screener_range_pagi_sore(
    n: int = Query(25, ge=5, le=60, description="Berapa kandidat likuid teratas yang dianalisis (default 25)"),
    jenis: str = Query("active", description="active (most active) | value (top value)"),
    min_range_persen: float = Query(None, ge=0, le=20, description="Rata-rata range harian (High-Low) minimal % (default 1.2)"),
    min_high_pagi: float = Query(None, ge=0, le=100, description="% hari dgn day-high di sesi pagi minimal (default 55)"),
    min_low_sore: float = Query(None, ge=0, le=100, description="% hari dgn day-low di sesi sore minimal (default 55)"),
    min_spread_bersih: float = Query(None, ge=0, le=10, description="Spread bersih minimal % setelah fee (default 0.3)")
):
    """
    SCREENER RANGE PAGI-SORE: cari saham PALING COCOK untuk strategi jual-pagi /
    beli-sore di antara kandidat likuid (nilai transaksi >= Rp5 miliar).

    Kriteria lolos: rata-rata range harian LEBAR (default >= 1.2%), day-high
    sering terjadi di pagi (>= 55%), day-low sering di sore (>= 25% — kalibrasi
    empiris: saham likuid BEI day-low-di-sore median 25%, max 36%), dan spread
    bersih setelah fee positif. Diurutkan dari spread bersih terbesar.

    Contoh: /v1/screener/range-pagi-sore?n=30&min_range_persen=1.5
    """
    data = screener_range_pagi_sore(
        n=n, jenis=jenis, min_range_persen=min_range_persen, min_high_pagi=min_high_pagi,
        min_low_sore=min_low_sore, min_spread_bersih=min_spread_bersih,
    )
    return {"status": "success", "data": data}


@router.get("/screener/range-pagi-sore/semua-saham")
async def run_screener_range_pagi_sore_semua_saham(
    offset: int = Query(0, ge=0),
    limit: int = Query(15, ge=1, le=30, description="Batas per batch (default 15 — tiap saham dianalisis penuh, cukup berat)"),
    min_range_persen: float = Query(None, ge=0, le=20),
    min_high_pagi: float = Query(None, ge=0, le=100),
    min_low_sore: float = Query(None, ge=0, le=100),
    min_spread_bersih: float = Query(None, ge=0, le=10)
):
    """
    Screener Range Pagi-Sore atas seluruh daftar BEI (dipaginasi). Lebih lambat
    dari versi default karena tiap saham dianalisis penuh (data intraday 15 menit).
    """
    total = len(SEMUA_SAHAM_BEI)
    slice_ticker = SEMUA_SAHAM_BEI[offset: offset + limit]
    if not slice_ticker:
        raise HTTPException(status_code=404, detail=f"Offset {offset} di luar jangkauan. Total saham di daftar BEI: {total}.")
    data = screener_range_pagi_sore(
        daftar_ticker=slice_ticker, min_range_persen=min_range_persen,
        min_high_pagi=min_high_pagi, min_low_sore=min_low_sore,
        min_spread_bersih=min_spread_bersih,
    )
    offset_berikutnya = offset + limit if (offset + limit) < total else None
    data["total_saham_di_daftar_bei"] = total
    data["diproses_offset"] = offset
    data["diproses_limit"] = limit
    data["offset_berikutnya"] = offset_berikutnya
    return {"status": "success", "data": data}


@router.get("/backtest/range-pagi-sore/{ticker}")
async def backtest_range_pagi_sore_endpoint(
    ticker: str,
    window_hari: int = Query(None, ge=10, le=120, description="Jendela hari walk-forward (default 30)"),
    persentil_jual: float = Query(None, gt=0, lt=1),
    persentil_beli: float = Query(None, gt=0, lt=1)
):
    """
    Backtest walk-forward strategi Range Pagi-Sore: level jual/beli tiap hari
    dihitung dari data hari-hari SEBELUMNYA (tanpa look-ahead), lalu dicek apakah
    jual pagi & beli sore terisi. Contoh: /v1/backtest/range-pagi-sore/BBRI
    """
    hasil = backtest_range_pagi_sore(
        ticker, window_hari=window_hari, persentil_jual=persentil_jual, persentil_beli=persentil_beli
    )
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data intraday untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


# =========================================================================
# STRATEGI 6: BPJS (Beli Pagi Jual Sore)
# =========================================================================

@router.get("/analisis/bpjs/{ticker}")
async def analisis_bpjs_saham(
    ticker: str,
    window_hari: int = Query(None, ge=10, le=120, description="Jendela hari statistik (default 20)"),
    persentil_beli: float = Query(None, gt=0, lt=1, description="Persentil trough pagi -> level beli (default 0.3)"),
    persentil_jual: float = Query(None, gt=0, lt=1, description="Persentil peak sore -> level jual (default 0.5)")
):
    """
    Analisis pola BPJS (Beli Pagi Jual Sore): rekomendasi harga pasang BELI di sesi
    pagi (morning low) dan JUAL di sesi sore (afternoon high) di HARI YANG SAMA,
    berdasarkan statistik N hari sebelumnya + estimasi peluang terisi.
    Contoh: /v1/analisis/bpjs/BBRI
    """
    data = hitung_analisis_bpjs(
        ticker, window_hari=window_hari, persentil_beli=persentil_beli, persentil_jual=persentil_jual
    )
    if not data:
        raise HTTPException(status_code=404, detail=f"Data intraday untuk {ticker.upper()} tidak cukup untuk analisis pola.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


# =========================================================================
# STRATEGI GABUNGAN: REKOMENDASI STRATEGI PALING COCOK UNTUK 1 SAHAM
# =========================================================================

@router.get("/analisis/strategi-gabungan/{ticker}")
async def analisis_strategi_gabungan(ticker: str):
    """
    Analisis GABUNGAN satu saham: semua strategi (swing-dividen, momentum gorengan,
    fast-intraday, BSJP, range pagi-sore, BPJS) dijalankan dari data yang sama,
    lalu diberikan skor kecocokan berdasarkan profil saham + sinyal aktif.
    Output: peringkat strategi, strategi terbaik, dan langkah eksekusi.
    Contoh: /v1/analisis/strategi-gabungan/BBRI
    """
    data = rekomendasi_strategi_gabungan(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak ditemukan atau tidak lengkap.")
    return {"status": "success", "data": _tambah_konteks_pasar(data, ticker)}


@router.get("/backtest/bpjs/{ticker}")
async def backtest_bpjs_endpoint(
    ticker: str,
    window_hari: int = Query(None, ge=10, le=120, description="Jendela hari walk-forward (default 20)"),
    persentil_beli: float = Query(None, gt=0, lt=1),
    persentil_jual: float = Query(None, gt=0, lt=1)
):
    """
    Backtest walk-forward strategi BPJS: level beli pagi & jual sore tiap hari
    dihitung dari data hari-hari SEBELUMNYA (tanpa look-ahead), lalu dicek apakah
    beli pagi & jual sore terisi. Contoh: /v1/backtest/bpjs/BBRI
    """
    hasil = backtest_bpjs(
        ticker, window_hari=window_hari, persentil_beli=persentil_beli, persentil_jual=persentil_jual
    )
    if not hasil:
        raise HTTPException(status_code=404, detail=f"Data intraday untuk {ticker.upper()} tidak cukup untuk backtest.")
    return {"status": "success", "data": hasil}


# =========================================================================
# VALIDASI: SKOR KECOCOKAN vs PERFORMA BACKTEST HISTORIS
# =========================================================================

@router.get("/backtest/validasi-strategi/{ticker}")
async def validasi_strategi_kecocokan(ticker: str):
    """
    Validasi skor kecocokan: bandingkan strategi terbaik versi skor (dari
    /analisis/strategi-gabungan) dengan strategi terbaik versi performa backtest
    historis (semua strategi dijalankan dari data historis). Output: peringkat
    skor vs peringkat backtest, cocok_top1, korelasi Spearman, dan catatan.

    Contoh: /v1/backtest/validasi-strategi/BBRI
    """
    data = validasi_kecocokan_satu_saham(ticker)
    if not data:
        raise HTTPException(status_code=404, detail=f"Data untuk {ticker.upper()} tidak cukup untuk validasi.")
    return {"status": "success", "data": data}


# =========================================================================
# REKOMENDASI PER SEKTOR: SAHAM TERBAIK PER STRATEGI DALAM SATU SEKTOR
# =========================================================================

@router.get("/rekomendasi/sektor")
async def rekomendasi_sektor_endpoint(
    industri: str = Query(..., description="Kata kunci industry Yahoo Finance, misal 'Bank', 'Coal', 'Real Estate', 'Software', 'Marine Shipping'"),
    top: int = Query(3, ge=1, le=5, description="Berapa saham terbaik per strategi (default 3)"),
    max_saham: int = Query(15, ge=1, le=60, description="Batas jumlah saham sektor yang diskor per request — Vercel serverless punya batas waktu eksekusi, sektor besar perlu dibatasi atau dipanggil berulang"),
    backtest: bool = Query(True, description="Jalankan backtest historis untuk finalis tiap strategi (false = lebih cepat)")
):
    """
    Rekomendasi saham TERBAIK per strategi dalam satu sektor/industri.

    Untuk tiap saham di sektor: jalankan rekomendasi_strategi_gabungan (skor
    kecocokan 6 strategi), lalu ranking per strategi, lalu (opsional) backtest
    historis untuk top-N finalis tiap strategi.

    Contoh:
      /v1/rekomendasi/sektor?industri=Bank
      /v1/rekomendasi/sektor?industri=Coal&top=3&backtest=false
      /v1/rekomendasi/sektor?industri=Software&max_saham=10&backtest=true
    """
    from app.rekomendasi_sektor import daftar_saham_sektor, rekomendasi_top_per_strategi

    tickers = daftar_saham_sektor(industri)
    if not tickers:
        raise HTTPException(
            status_code=404,
            detail=f"Tidak ada saham dengan industry mengandung '{industri}'. Coba: Bank, Coal, Real Estate, Software, Marine Shipping, Oil & Gas, dll."
        )

    total_sektor = len(tickers)
    dibatasi = total_sektor > max_saham
    if dibatasi:
        tickers = tickers[:max_saham]

    # max_workers dikurangi (vs screener lain) karena tiap saham memicu BANYAK
    # request ke Yahoo (history + intraday + info) — paralelisme tinggi di IP
    # serverless cepat kena rate-limit Yahoo (429). 3 worker + retry backoff di
    # rekomendasi_top_per_strategi jauh lebih tahan.
    hasil = rekomendasi_top_per_strategi(
        tickers, top_n=top, max_workers=min(MAX_WORKERS_SCREENER, 3), backtest=backtest
    )

    if hasil['jumlah_saham_berhasil'] == 0 and hasil['saham_gagal']:
        peringatan = (
            "Semua saham gagal dianalisis — kemungkinan besar rate-limit data Yahoo "
            "pada IP server. Coba lagi 1-2 menit kemudian, atau kurangi max_saham "
            "(contoh: max_saham=5) untuk memangkas beban request."
        )
    else:
        peringatan = None

    return {
        "status": "success",
        "data": {
            "kata_kunci": industri,
            "jumlah_saham_di_sektor": total_sektor,
            "jumlah_saham_diskor": len(tickers),
            "keterangan_batas": (
                f"Sektor '{industri}' punya {total_sektor} saham; request ini diskor "
                f"{len(tickers)} saham pertama (max_saham={max_saham}). Naikkan max_saham "
                "untuk cakupan penuh."
                if dibatasi else None
            ),
            "peringatan": peringatan,
            **hasil,
        }
    }


# =========================================================================
# MOST ACTIVE / TOP VALUE / TOP GAINER + REKOMENDASI INTRADAY
# =========================================================================

@router.get("/pasar/most-active")
async def pasar_most_active(
    jenis: str = Query("active", description="active | value | gainer | loser"),
    limit: int = Query(10, ge=1, le=30)
):
    """
    Dashboard pasar: top saham berdasarkan metrik likuiditas/pergerakan, dihitung
    SENDIRI dari batch download seluruh daftar BEI (IDX API diblokir Cloudflare,
    Yahoo screener rate-limited — jadi metrik dihitung dari data yang sama dengan
    strategi lain di sistem ini).

    jenis:
      active = Most Active (volume lembar terbesar)
      value  = Top Value (nilai transaksi harga x volume terbesar)
      gainer = Top Gainer (kenaikan % harian tertinggi)
      loser  = Top Loser (penurunan % harian terbesar)

    CATATAN: Top Gainer TIDAK untuk dibeli (uji empiris: cenderung reversal
    di T+1). Most Active cocok sebagai FILTER likuiditas, bukan sinyal arah.

    Contoh: /v1/pasar/most-active?jenis=active&limit=10
    """
    try:
        data = top_pasar(jenis=jenis, limit=limit)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "success", "data": data}


@router.get("/rekomendasi/intraday")
async def rekomendasi_intraday(
    n: int = Query(5, ge=1, le=15, description="Berapa kandidat most-active yang diskor strategi intraday (default 5 — batch pasar + analisis gabungan butuh waktu; n=15 bisa dekat batas timeout serverless)"),
    jenis: str = Query("active", description="active (most active) | value (top value)"),
    eksekusi: bool = Query(False, description="true = one-click: sertakan detail eksekusi (harga pasang order / entry-TP-SL) utk top pick tiap strategi intraday")
):
    """
    Rekomendasi saham UNTUK STRATEGI INTRADAY berbasis filter likuiditas.

    Alur: batch download seluruh BEI -> ambil top-n saham PALING LIKUID (most
    active / top value, min nilai transaksi Rp5 miliar) -> jalankan skor
    kecocokan 6 strategi untuk tiap kandidat -> ranking per strategi intraday
    (BPJS, BSJP, range-pagi-sore, fast-intraday).

    Most Active dipakai sebagai FILTER LIKUIDITAS saja — arah keputusan tetap
    dari sinyal strategi (top gainer tidak dipakai sebagai sinyal beli: empiris
    cenderung reversal). Untuk eksekusi spesifik, jalankan analisis per saham:
    /v1/analisis/bpjs/{ticker}, /v1/analisis/bsjp/{ticker}, dst.

    Contoh: /v1/rekomendasi/intraday?n=8
    """
    if jenis not in ("active", "value"):
        raise HTTPException(status_code=400, detail="jenis hanya mendukung: active | value")
    # Batas waktu 150 dtk utk fase skor + eksekusi: batch pasar di-cache harian
    # (cepat), tapi skor strategi tetap diberi anggaran supaya tidak pernah 504.
    data = rekomendasi_intraday_likuid(n=n, jenis=jenis, batas_waktu_detik=150, eksekusi=eksekusi)
    return {"status": "success", "data": data}


@router.get("/pasar/digest-pagi")
async def pasar_digest_pagi(
    limit_active: int = Query(10, ge=1, le=30, description="Berapa saham most-active yang ditampilkan"),
    n_intraday: int = Query(5, ge=1, le=15, description="Berapa kandidat likuid yang diskor strategi intraday"),
    jenis: str = Query("active", description="active (most active) | value (top value)"),
    skor_budget_detik: int = Query(150, ge=30, le=200, description="Anggaran waktu (dtk) utk fase skor strategi intraday — kalau habis, hasil parsial dikembalikan (bukan timeout). Maks 200 supaya masih ada ruang untuk batch pasar di maxDuration 300s")
):
    """
    DIGEST PAGI: Most Active + Rekomendasi Intraday dalam SATU request.

    Dibuat untuk notifikasi WA otomatis tiap pagi — batch download 941 saham BEI
    dijalankan SEKALI (di-cache per sesi WIB) lalu dipakai bersama untuk dua
    output (hemat waktu & mengurangi risiko rate-limit Yahoo). Data = hari
    trading lengkap terakhir.

    Contoh: /v1/pasar/digest-pagi?limit_active=10&n_intraday=5
    """
    if jenis not in ("active", "value"):
        raise HTTPException(status_code=400, detail="jenis hanya mendukung: active | value")
    data = digest_pagi(
        limit_active=limit_active, n_intraday=n_intraday, jenis=jenis,
        skor_budget_detik=skor_budget_detik,
    )
    return {"status": "success", "data": data}


@router.get("/pasar/riwayat-digest")
async def pasar_riwayat_digest(
    hari: int = Query(7, ge=1, le=30, description="Berapa hari riwayat digest yang diambil (terbaru dulu)")
):
    """
    Riwayat digest harian: snapshot Most Active + Rekomendasi Intraday yang
    tersimpan otomatis setiap kali /v1/pasar/digest-pagi dipanggil.

    Penyimpanan memakai Upstash Redis (KV serverless). Kalau env var
    UPSTASH_REDIS_REST_URL / UPSTASH_REDIS_REST_TOKEN belum dipasang, endpoint
    tetap jalan dengan riwayat kosong + peringatan (fitur riwayat nonaktif).

    Contoh: /v1/pasar/riwayat-digest?hari=7
    """
    data = ambil_riwayat_digest(hari=hari)
    return {"status": "success", "data": data}


@router.get("/rekomendasi/ringkasan-mingguan")
async def rekomendasi_ringkasan_mingguan(
    hari: int = Query(7, ge=1, le=30, description="Berapa hari riwayat yang diringkas"),
    n_saham: int = Query(3, ge=1, le=6, description="Berapa saham teratas yang dibacktest"),
    backtest: bool = Query(True, description="Jalankan backtest 5 strategi intraday (false = recap riwayat saja, lebih cepat)")
):
    """
    RINGKASAN MINGGUAN: recap riwayat digest 7 hari + performa backtest 5 strategi
    intraday (BPJS, BSJP, range-pagi-sore, fast-intraday, gorengan) pada saham
    teratas. Dipakai notifikasi WA mingguan (Minggu sore).

    Contoh: /v1/rekomendasi/ringkasan-mingguan?hari=7&n_saham=3&backtest=true
    """
    from app.ringkasan_mingguan import ringkasan_mingguan
    data = ringkasan_mingguan(hari=hari, n_saham=n_saham, backtest=backtest)
    return {"status": "success", "data": data}
