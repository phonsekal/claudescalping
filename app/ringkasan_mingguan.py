# app/ringkasan_mingguan.py
"""
RINGKASAN MINGGUAN: recap riwayat digest 7 hari + performa backtest 5 strategi
intraday pada saham-saham teratas.

Dua sumber data:
1. Riwayat digest harian (Upstash KV, opsional) -> recap_mingguan:
   saham paling sering muncul di Most Active + statistik per strategi intraday
   (hari sinyal aktif, skor rata-rata, top picks) selama N hari terakhir.
2. Backtest historis 5 strategi intraday (BPJS, BSJP, range-pagi-sore,
   fast-intraday, gorengan) pada saham-saham paling sering muncul -> performa
   nyata versi walk-forward (win rate, return, ekspektasi per hari).

Kalau KV belum dikonfigurasi, bagian recap memakai riwayat kosong (tidak error),
dan backtest tetap dijalankan pada watchlist likuid sebagai pengganti.

CATATAN JUJUR: backtest intraday pakai data 60 hari (batas yfinance), jadi jumlah
sampel terbatas — anggap sanity-check, bukan validasi final. Sama seperti catatan
di backtest.py.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.riwayat import ambil_riwayat_digest, recap_dari_riwayat
from app.backtest import (
    backtest_bpjs, backtest_bsjp, backtest_range_pagi_sore,
    backtest_fast_intraday, backtest_gorengan_momentum,
)
from app.config import WATCHLIST_FAST_INTRADAY

# 5 strategi intraday yang dibacktest + label + tipe agregasi.
STRATEGI_INTRADAY = [
    ("bpjs", "BPJS (Beli Pagi Jual Sore)", backtest_bpjs, "per_hari"),
    ("bsjp", "BSJP (Beli Sore Jual Pagi)", backtest_bsjp, "per_transaksi"),
    ("range_pagi_sore", "Range Pagi-Sore (Jual Pagi, Beli Sore)", backtest_range_pagi_sore, "per_hari"),
    ("fast_intraday", "Fast Intraday Alert (15 menit)", backtest_fast_intraday, "per_transaksi"),
    ("gorengan", "Momentum Gorengan (Day Trading)", backtest_gorengan_momentum, "per_transaksi"),
]


def _backtest_satu_strategi(ticker_list, fungsi, tipe, max_workers=3):
    """Jalankan backtest satu strategi untuk banyak ticker, gabungkan hasilnya."""
    hasil_per_saham = []

    def proses(t):
        try:
            return fungsi(t)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in ticker_list]
        for future in as_completed(futures):
            hasil_per_saham.append(future.result())

    # Filter per tipe: hasil per_hari (BPJS/Range) TIDAK punya field total_transaksi
    # (mereka memakai jumlah_roundtrip/jumlah_hari_diuji) — filter harus type-aware,
    # kalau tidak hasil per_hari selalu dianggap kosong (bug: 0 hari diuji padahal
    # data ada).
    if tipe == "per_hari":
        valid = [h for h in hasil_per_saham
                 if h and h.get("jumlah_hari_diuji", 0) and h.get("jumlah_hari_diuji", 0) > 0]
    else:
        valid = [h for h in hasil_per_saham if h and h.get("total_transaksi", 0) > 0]

    if tipe == "per_hari":
        # BPJS & Range Pagi-Sore: hasil berbasis hari (bukan per transaksi).
        hari_total = sum(h.get("jumlah_hari_diuji", 0) for h in valid)
        roundtrip_total = sum(h.get("jumlah_roundtrip", 0) for h in valid)
        ekspektasi = [h.get("ekspektasi_per_hari_persen") for h in valid if h.get("ekspektasi_per_hari_persen") is not None]
        return {
            "tipe": "per_hari",
            "jumlah_saham_diuji": len(valid),
            "total_hari_diuji": hari_total,
            "total_roundtrip": roundtrip_total,
            "persen_hari_roundtrip_rata2": round(
                sum(h.get("persen_hari_roundtrip_terisi", 0) for h in valid) / len(valid), 1
            ) if valid else None,
            "ekspektasi_per_hari_rata2": round(sum(ekspektasi) / len(ekspektasi), 3) if ekspektasi else None,
            "saham_terbaik": max(
                ({"saham": h["saham"], "ekspektasi_per_hari_persen": h.get("ekspektasi_per_hari_persen"),
                  "jumlah_roundtrip": h.get("jumlah_roundtrip", 0)} for h in valid),
                key=lambda x: x["ekspektasi_per_hari_persen"] or -999,
                default=None,
            ),
        }

    # Tipe per_transaksi (BSJP, Fast Intraday, Gorengan)
    total_transaksi = sum(h.get("total_transaksi", 0) for h in valid)
    if not valid:
        return {"tipe": "per_transaksi", "jumlah_saham_diuji": 0, "total_transaksi": 0}

    win_rate = sum(h.get("win_rate_persen", 0) * h.get("total_transaksi", 0) for h in valid) / total_transaksi
    rata_return = sum(h.get("rata_rata_return_per_transaksi_persen", 0) * h.get("total_transaksi", 0) for h in valid) / total_transaksi
    max_dd = min(h.get("max_drawdown_persen", 0) for h in valid)
    return {
        "tipe": "per_transaksi",
        "jumlah_saham_diuji": len(valid),
        "total_transaksi": total_transaksi,
        "win_rate_gabungan_persen": round(win_rate, 2),
        "rata_rata_return_per_transaksi_persen": round(rata_return, 2),
        "max_drawdown_terburuk_persen": max_dd,
        "saham_terbaik": max(
            ({"saham": h["saham"], "total_transaksi": h.get("total_transaksi", 0),
              "win_rate_persen": h.get("win_rate_persen"),
              "total_return_gabungan_persen": h.get("total_return_gabungan_persen")} for h in valid),
            key=lambda x: x.get("total_return_gabungan_persen") or -999,
            default=None,
        ),
    }


def ringkasan_mingguan(hari: int = 7, n_saham: int = 3, backtest: bool = True,
                       max_workers: int = 3) -> dict:
    """
    Ringkasan mingguan: recap riwayat digest + (opsional) backtest 5 strategi
    intraday pada saham-saham teratas.

    Parameter:
      hari      - berapa hari riwayat yang diringkas (default 7)
      n_saham   - berapa saham teratas yang dipakai untuk backtest (default 3)
      backtest  - jalankan backtest historis (default True)
    """
    riwayat_data = ambil_riwayat_digest(hari=hari)
    riwayat = riwayat_data.get("riwayat", [])
    recap = recap_dari_riwayat(riwayat)

    tanggal_list = [r.get("tanggal") for r in riwayat if r.get("tanggal")]
    periode = {
        "jumlah_hari_riwayat": len(riwayat),
        "tanggal_awal": min(tanggal_list) if tanggal_list else None,
        "tanggal_akhir": max(tanggal_list) if tanggal_list else None,
    }

    # Pilih saham untuk backtest: dari recap (paling sering muncul), fallback watchlist likuid.
    ticker_backtest = [s["ticker"] for s in recap["saham_teratas_most_active"][:n_saham]]
    sumber_ticker = "recap_most_active"
    if not ticker_backtest:
        ticker_backtest = [t.replace(".JK", "") for t in WATCHLIST_FAST_INTRADAY[:n_saham]]
        sumber_ticker = "watchlist_likuid_fallback"

    performa = []
    if backtest and ticker_backtest:
        for kode, nama, fungsi, tipe in STRATEGI_INTRADAY:
            hasil = _backtest_satu_strategi(ticker_backtest, fungsi, tipe, max_workers=max_workers)
            hasil["kode"] = kode
            hasil["nama"] = nama
            performa.append(hasil)

    return {
        "periode": periode,
        "recap_mingguan": recap,
        "kv_terkonfigurasi": riwayat_data.get("tersedia", False),
        "peringatan_riwayat": riwayat_data.get("peringatan"),
        "backtest": {
            "dijalankan": bool(backtest and ticker_backtest),
            "sumber_saham": sumber_ticker,
            "saham_diuji": ticker_backtest,
            "per_strategi": performa,
            "catatan": (
                "Backtest intraday memakai data 60 hari terakhir (batas yfinance) — "
                "jumlah sampel terbatas, anggap sanity-check bukan validasi final."
                if backtest else None
            ),
        },
    }
