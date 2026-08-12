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
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd

from app.daftar_saham_bei import SEMUA_SAHAM_BEI
from app.services import (
    rekomendasi_strategi_gabungan,
    hitung_analisis_bpjs,
    hitung_analisis_range_pagi_sore,
    hitung_sinyal_bsjp,
    hitung_sinyal_fast_intraday,
)
from app.riwayat import simpan_snapshot_digest, simpan_kv_json, ambil_kv_json

# TTL cache batch pasar (detik). 48 jam cukup: kunci dipisah per hari WIB + sesi
# (pagi < 16:00 pakai hari trading lengkap sebelumnya, sore >= 16:00 pakai hari
# berjalan), jadi TTL cuma berfungsi membersihkan key lama.
CACHE_PASAR_TTL_DETIK = 172800  # 48 jam

# Anggaran waktu (detik) untuk fase skor strategi intraday di digest — supaya
# request TIDAK pernah 504 walaupun Yahoo sedang rate-limit: kalau budget habis,
# digest tetap balik dengan hasil parsial + daftar saham yang belum diproses.
DIGEST_SKOR_BUDGET_DETIK = 150

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


def _peta_peringkat(metrik: dict, kunci: str, reverse: bool) -> dict:
    """
    Peta {ticker: peringkat} untuk satu metrik (volume, nilai, % hari).
    Digunakan untuk badge konteks pasar per saham.
    """
    items = [mm for mm in metrik.values() if mm.get(kunci) is not None]
    items.sort(key=lambda x: x[kunci], reverse=reverse)
    return {mm["ticker"]: i + 1 for i, mm in enumerate(items)}


def _kunci_cache_pasar() -> str:
    """
    Kunci cache batch pasar untuk HARI INI (WIB) + sesi.

    Sesi dibedakan karena metrik berubah setelah jam 16:00 WIB: sebelum itu data
    = hari trading lengkap terakhir (hari sebelumnya), sesudahnya = hari ini.
    """
    now = datetime.now(ZoneInfo("Asia/Jakarta"))
    sesi = "pagi" if now.hour < 16 else "sore"
    return f"metrik_pasar:{now.strftime('%Y-%m-%d')}:{sesi}"


def ambil_metrik_pasar() -> dict:
    """
    Batch download seluruh daftar BEI (5d, harian) dan hitung metrik pasar.

    Return: {"metrik": {ticker: {...}}, "tanggal_data": ..., "jumlah_berhasil": n}
    Retry 1x per chunk untuk menyerap rate-limit Yahoo sesaat.

    CACHE OPSIONAL (Upstash KV): dalam satu sesi WIB, metrik pasar TIDAK
    berubah — jadi panggilan ulang (most-active, intraday, digest) memakai
    hasil batch yang sama tanpa download 941 saham berulang. Cache hanya
    disimpan kalau batch LENGKAP (tanpa chunk gagal); kalau parsial, tidak
    dicache supaya panggilan berikutnya mencoba batch penuh lagi.
    """
    kunci = _kunci_cache_pasar()
    cached = ambil_kv_json(kunci)
    if cached:
        return cached

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

    hasil = {
        "metrik": metrik,
        "tanggal_data": tanggal,
        "jumlah_berhasil": len(metrik),
        "chunk_gagal": chunk_gagal,
    }
    if not chunk_gagal and metrik:
        # Batch lengkap -> cache untuk sesi ini (hemat download berulang).
        simpan_kv_json(kunci, hasil, ex_detik=CACHE_PASAR_TTL_DETIK)
    return hasil


def top_pasar(jenis: str = "active", limit: int = 10, metrik_pasar: dict = None) -> dict:
    """
    Dashboard pasar: top saham per kategori metrik.

    jenis: active (volume terbesar) | value (nilai transaksi terbesar) |
           gainer (perubahan % tertinggi) | loser (perubahan % terendah)

    metrik_pasar (opsional): hasil ambil_metrik_pasar() yang sudah di-fetch,
    dipakai digest_pagi() supaya batch download 941 saham hanya terjadi SEKALI
    untuk beberapa output (hemat waktu + anti rate-limit).
    """
    hasil = metrik_pasar if metrik_pasar is not None else ambil_metrik_pasar()
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

    # Badge konteks per item: peringkat di tiap kategori pasar + kelayakan
    # intraday (nilai transaksi >= Rp5 miliar). Dipakai formatter WA utk
    # menampilkan posisi gainer/loser dan nilai transaksi di daftar pasar.
    p_aktif = _peta_peringkat(metrik, "volume", True)
    p_nilai = _peta_peringkat(metrik, "nilai_transaksi", True)
    p_gainer = _peta_peringkat(metrik, "pct_change_hari", True)
    p_loser = _peta_peringkat(metrik, "pct_change_hari", False)
    for m in daftar:
        t = m["ticker"]
        m["peringkat_aktif"] = p_aktif.get(t)
        m["peringkat_nilai"] = p_nilai.get(t)
        m["peringkat_gainer"] = p_gainer.get(t)
        m["peringkat_loser"] = p_loser.get(t)
        m["layak_intraday"] = m["nilai_transaksi"] >= MIN_NILAI_TRANSAKSI_INTRADAY

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


def _buat_konteks(t: str, m: dict, metrik: dict, tanggal: str) -> dict:
    """Susun badge konteks pasar untuk satu saham dari metrik batch pasar."""
    p_aktif = _peta_peringkat(metrik, "volume", True)
    p_nilai = _peta_peringkat(metrik, "nilai_transaksi", True)
    p_gainer = _peta_peringkat(metrik, "pct_change_hari", True)
    p_loser = _peta_peringkat(metrik, "pct_change_hari", False)
    pos_aktif = p_aktif.get(t)
    pos_nilai = p_nilai.get(t)
    pos_gainer = p_gainer.get(t)
    pos_loser = p_loser.get(t)
    layak = m["nilai_transaksi"] >= MIN_NILAI_TRANSAKSI_INTRADAY

    label_parts = []
    if pos_aktif:
        label_parts.append(f"🔥 Most Active #{pos_aktif}")
    if pos_nilai:
        label_parts.append(f"💎 Top Value #{pos_nilai}")
    if pos_gainer and pos_gainer <= 20:
        label_parts.append(f"🚀 Top Gainer #{pos_gainer}")
    if pos_loser and pos_loser <= 20:
        label_parts.append(f"📉 Top Loser #{pos_loser}")
    label = " · ".join(label_parts) if label_parts else "🌐 Saham di luar radar pasar"
    label += f" | {'💧 Likuid intraday' if layak else '⚠️ Likuiditas rendah utk intraday'}"

    return {
        "tersedia": True,
        "ticker": t,
        "tanggal_data": tanggal,
        "harga": m["harga"],
        "pct_change_hari": m.get("pct_change_hari"),
        "nilai_transaksi_miliar": round(m["nilai_transaksi"] / 1e9, 1),
        "layak_intraday": layak,
        "posisi_most_active": pos_aktif,
        "posisi_top_value": pos_nilai,
        "posisi_gainer": pos_gainer,
        "posisi_loser": pos_loser,
        "label": label,
    }


def konteks_pasar_saham(ticker: str):
    """
    BADGE konteks pasar untuk satu saham (dipakai di analisis per saham).

    Sumber utama: cache batch pasar (cepat — KV Upstash). Kalau cache belum ada,
    fallback menghitung metrik SATU saham saja (tanpa batch 941) supaya analisis
    saham tidak jadi lambat; posisi pasar (peringkat) dikosongkan.
    """
    t = _nama_bisa(str(ticker).upper())
    cached = ambil_kv_json(_kunci_cache_pasar())
    if cached and cached.get("metrik"):
        metrik = cached["metrik"]
        m = metrik.get(t) or metrik.get(f"{t}.JK")
        if m:
            return _buat_konteks(t, m, metrik, cached.get("tanggal_data"))
        return {"tersedia": False, "ticker": t, "alasan": "tidak ada di batch pasar terakhir (mungkin baru listing / delisting)"}
    try:
        df = yf.download(f"{t}.JK", period=BATCH_PERIODE, interval=BATCH_INTERVAL,
                         auto_adjust=False, progress=False)
        m = _ekstrak_metrik(f"{t}.JK", df)
        if not m:
            return {"tersedia": False, "ticker": t, "alasan": "data pasar tidak tersedia"}
        layak = m["nilai_transaksi"] >= MIN_NILAI_TRANSAKSI_INTRADAY
        return {
            "tersedia": True,
            "ticker": t,
            "tanggal_data": m.get("tanggal"),
            "harga": m["harga"],
            "pct_change_hari": m.get("pct_change_hari"),
            "nilai_transaksi_miliar": round(m["nilai_transaksi"] / 1e9, 1),
            "layak_intraday": layak,
            "posisi_most_active": None,
            "posisi_top_value": None,
            "posisi_gainer": None,
            "posisi_loser": None,
            "label": ("💧 Likuid intraday" if layak else "⚠️ Likuiditas rendah utk intraday")
                      + " (posisi pasar belum tersedia — batch pasar hari ini belum dihitung)",
        }
    except Exception:
        return {"tersedia": False, "ticker": t, "alasan": "gagal ambil data pasar"}


def _hitung_eksekusi(top_pick: dict, deadline=None) -> dict:
    """
    ONE-CLICK: jalankan analisis strategi-spesifik utk top pick tiap strategi
    intraday dan kumpulkan detail eksekusi (harga pasang order / entry/TP/SL).
    Menghormati deadline supaya tidak pernah 504.
    """
    fn_map = {
        "bpjs": lambda t: hitung_analisis_bpjs(t),
        "range_pagi_sore": lambda t: hitung_analisis_range_pagi_sore(t),
        "bsjp": lambda t: hitung_sinyal_bsjp(t),
        "fast_intraday": lambda t: hitung_sinyal_fast_intraday(t),
    }
    hasil = {}
    if not top_pick:
        return hasil
    ex = ThreadPoolExecutor(max_workers=min(4, len(top_pick)))
    try:
        futures = {}
        for kode, (t, nama) in top_pick.items():
            if deadline is not None and time.time() >= deadline:
                hasil[kode] = {"nama": nama, "ticker": t, "detail": None,
                               "status": "belum sempat dianalisis (budget habis)"}
                continue
            futures[ex.submit(fn_map[kode], t)] = (kode, t, nama)
            time.sleep(0.3)
        if futures:
            sisa = None if deadline is None else max(0.0, deadline - time.time())
            try:
                for fut in as_completed(futures, timeout=sisa):
                    kode, t, nama = futures[fut]
                    try:
                        det = fut.result()
                    except Exception:
                        det = None
                    hasil[kode] = {"nama": nama, "ticker": t, "detail": det}
            except FuturesTimeout:
                for fut, (kode, t, nama) in futures.items():
                    if kode not in hasil:
                        hasil[kode] = {"nama": nama, "ticker": t, "detail": None,
                                       "status": "belum selesai (budget habis)"}
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return hasil


def rekomendasi_intraday_likuid(n: int = 8, jenis: str = "active", max_workers: int = 3,
                                metrik_pasar: dict = None, batas_waktu_detik: int = None,
                                eksekusi: bool = False) -> dict:
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

    metrik_pasar (opsional): hasil ambil_metrik_pasar() yang sudah di-fetch,
    dipakai digest_pagi() supaya batch pasar tidak di-download dua kali.
    """
    pasar = metrik_pasar if metrik_pasar is not None else ambil_metrik_pasar()
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

    def _jalankan_batch(daftar, worker, deadline=None):
        """
        Jalankan skor strategi untuk daftar ticker (paralel, spacing anti-429).

        deadline (opsional): timestamp monolitik. Kalau sudah lewat deadline,
        berhenti submit tugas baru dan hanya kumpulkan yang sudah selesai —
        sisanya masuk daftar "belum diproses" (gag). Dipakai supaya request
        serverless TIDAK pernah 504 walau Yahoo lambat/rate-limit.
        """
        ok = {}
        gag = []
        if not daftar:
            return ok, gag
        ex = ThreadPoolExecutor(max_workers=worker)
        try:
            futures = {}
            for t in daftar:
                if deadline is not None and time.time() >= deadline:
                    gag.append(t)  # tidak sempat diproses
                    continue
                futures[ex.submit(_proses, t)] = t
                time.sleep(0.4)  # spacing antar submit, anti burst 429
            if futures:
                sisa = None if deadline is None else max(0.0, deadline - time.time())
                try:
                    for fut in as_completed(futures, timeout=sisa):
                        t = futures[fut]
                        try:
                            _, r = fut.result()  # _proses mengembalikan (ticker, hasil)
                        except Exception:
                            r = None
                        if r:
                            ok[t] = r
                        else:
                            gag.append(t)
                except FuturesTimeout:
                    # Budget habis: tugas yang belum selesai = belum diproses.
                    for fut, t in futures.items():
                        if t not in ok and t not in gag:
                            gag.append(t)
        finally:
            # JANGAN menunggu tugas yang masih jalan (tidak boleh menggagalkan
            # request yang sudah punya hasil parsial).
            ex.shutdown(wait=False, cancel_futures=True)
        return ok, gag

    deadline = None
    if batas_waktu_detik:
        deadline = time.time() + batas_waktu_detik

    gabungan = {}
    ok1, gagal = _jalankan_batch(ticker_list, max_workers, deadline=deadline)
    gabungan.update(ok1)
    if gagal and len(gabungan) < len(ticker_list):
        if deadline is None or time.time() < deadline:
            time.sleep(2.0)
            ok2, gagal = _jalankan_batch(gagal, max(1, min(max_workers, 2)), deadline=deadline)
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
    top_pick = {}
    for kode, nama in STRATEGI_INTRADAY:
        lst = sorted(skor_by[kode], key=lambda x: -x["skor"])
        leaderboard.append({"kode": kode, "nama": nama, "terbaik": lst[:3]})
        if eksekusi and lst:
            top_pick[kode] = (lst[0]["ticker"], nama)  # utk one-click eksekusi

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

    resp = {
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
    if eksekusi:
        # ONE-CLICK: detail eksekusi top pick tiap strategi (dalam budget waktu
        # yang sama supaya request tidak 504 saat Yahoo lambat).
        resp["eksekusi"] = _hitung_eksekusi(top_pick, deadline)
    return resp


def _snapshot_ringkas(payload: dict) -> dict:
    """Buat snapshot ringkas digest untuk riwayat harian (hemat storage KV)."""
    return {
        "tanggal": payload.get("tanggal_data"),
        "jumlah_saham_dianalisis": payload.get("jumlah_saham_dianalisis"),
        "most_active": [
            {
                "ticker": m["ticker"],
                "harga": m["harga"],
                "pct_change_hari": m.get("pct_change_hari"),
                "nilai_transaksi_miliar": round(m["nilai_transaksi"] / 1e9, 1),
            }
            for m in (payload.get("most_active") or {}).get("list", [])[:10]
        ],
        "rekomendasi_intraday": [
            {
                "kode": lb["kode"],
                "nama": lb["nama"],
                "terbaik": [
                    {
                        "ticker": f["ticker"],
                        "skor": f["skor"],
                        "sinyal_aktif": f.get("sinyal_aktif"),
                    }
                    for f in lb.get("terbaik", [])
                ],
            }
            for lb in (payload.get("rekomendasi_intraday") or {})
            .get("leaderboard_per_strategi_intraday", [])
        ],
        # Penanda kejujuran: kalau budget skor habis, catat saham yang belum
        # diproses supaya recap mingguan tahu snapshot ini parsial.
        "saham_skor_belum_diproses": (payload.get("rekomendasi_intraday") or {}).get("saham_gagal", []),
    }


def digest_pagi(limit_active: int = 10, n_intraday: int = 5, jenis: str = "active",
                max_workers: int = 3, skor_budget_detik: int = DIGEST_SKOR_BUDGET_DETIK) -> dict:
    """
    DIGEST PAGI: Most Active + Rekomendasi Intraday dalam SATU request.

    Dipakai notifikasi WA otomatis tiap pagi. Kunci efisiensi: batch download
    seluruh BEI (941 ticker) hanya dijalankan SEKALI (lalu di-cache ke Upstash
    per sesi WIB), dan dipakai bersama oleh top_pasar() dan
    rekomendasi_intraday_likuid() via param metrik_pasar.

    skor_budget_detik: anggaran waktu untuk fase skor strategi intraday.
    Kalau habis, hasil parsial dikembalikan (bukan 504) + daftar saham yang
    belum diproses. Aman diset besar karena batch pasar sudah di-cache.

    Snapshot ringkas otomatis disimpan ke riwayat harian (Upstash KV, OPSIONAL):
    kalau KV belum dikonfigurasi, penyimpanan di-skip tanpa error.
    """
    pasar = ambil_metrik_pasar()
    most_active = top_pasar(jenis="active", limit=limit_active, metrik_pasar=pasar)
    intraday = rekomendasi_intraday_likuid(
        n=n_intraday, jenis=jenis, max_workers=max_workers, metrik_pasar=pasar,
        batas_waktu_detik=skor_budget_detik,
    )
    payload = {
        "tanggal_data": pasar.get("tanggal_data"),
        "jumlah_saham_dianalisis": pasar.get("jumlah_berhasil"),
        "chunk_gagal": pasar.get("chunk_gagal") or [],
        "most_active": most_active,
        "rekomendasi_intraday": intraday,
    }
    # Simpan riwayat (best-effort — kegagalan KV tidak boleh menggagalkan digest).
    try:
        payload["riwayat_tersimpan"] = simpan_snapshot_digest(_snapshot_ringkas(payload))
    except Exception:
        payload["riwayat_tersimpan"] = {"tersimpan": False, "alasan": "error saat simpan riwayat"}
    return payload
