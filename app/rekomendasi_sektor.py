# app/rekomendasi_sektor.py
"""
Rekomendasi saham TERBAIK per strategi dalam SATU sektor/industri.

Alur:
1. Ambil daftar ticker sektor dari cache sektor (audit_bei_sektor_cache.jsonl).
   Kalau cache belum ada, fallback: scan seluruh daftar BEI via Yahoo Finance
   (lambat, ~5 menit) lalu isi cache-nya.
2. Untuk tiap saham: jalankan rekomendasi_strategi_gabungan (skor kecocokan
   semua strategi dari data yang ditarik SEKALI — profil fit + sinyal aktif).
3. Ranking per strategi: saham dengan skor tertinggi = paling cocok untuk
   strategi itu saat ini.
4. Untuk top-N finalis tiap strategi: jalankan backtest historis strategi tsb
   untuk menampilkan performa lampau (win rate, return, sampel, flag andal).

Contoh:
    python -m app.rekomendasi_sektor --industri=Bank --top=3
    python -m app.rekomendasi_sektor --industri=Coal --top=3 --max-workers=6

Catatan: skor mengukur 'kecocokan profil + sinyal SEKARANG'. Backtest finalis
menampilkan performa historis sebagai sanity-check — perhatikan flag 'andal'
(sampel kecil = belum statistik kuat, terutama strategi intraday ~60 hari data).
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services import rekomendasi_strategi_gabungan
from app.validasi_strategi import STRATEGI_BACKTEST, NAMA_STRATEGI, _metrik_per_hari

CACHE_SEKTOR = "audit_bei_sektor_cache.jsonl"


def daftar_saham_sektor(kata_kunci: str) -> list:
    """
    Ticker saham yang industry Yahoo-nya mengandung kata_kunci (case-insensitive).

    Prioritas: cache sektor (file JSONL hasil audit 941 saham) — instan.
    Fallback kalau cache tidak ada: scan SEMUA_SAHAM_BEI via Yahoo .info
    (butuh ~5 menit), dan cache hasilnya supaya run berikutnya instan.
    """
    hasil = []
    cache_ada = os.path.exists(CACHE_SEKTOR)
    if cache_ada:
        with open(CACHE_SEKTOR) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for t, v in json.loads(line).items():
                    ind = (v or {}).get('industry') or ''
                    if kata_kunci.lower() in ind.lower():
                        hasil.append(t)
        if hasil:
            return sorted(hasil)

    # Cache tidak ada / tidak ada yang cocok -> scan seluruh BEI via Yahoo.
    from app.daftar_saham_bei import SEMUA_SAHAM_BEI
    import yfinance as yf

    baru = {}

    def cek(t):
        try:
            info = yf.Ticker(t).info
            return t, info.get('industry') or ''
        except Exception:
            return t, ''

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(cek, t) for t in SEMUA_SAHAM_BEI]
        for fut in as_completed(futures):
            t, ind = fut.result()
            baru[t] = {'sector': None, 'industry': ind}
            if kata_kunci.lower() in ind.lower():
                hasil.append(t)

    # PENTING: jangan menimpa cache yang SUDAH ADA tapi kebetulan tidak cocok
    # dengan kata kunci — itu cache hasil audit sektor yang punya data 'sector'
    # lengkap. Hanya tulis kalau cache belum pernah ada sama sekali.
    if not cache_ada:
        with open(CACHE_SEKTOR, 'w') as f:
            for t, v in baru.items():
                f.write(json.dumps({t: v}, ensure_ascii=False) + "\n")
    return sorted(hasil)


def _proses_satu_saham(t):
    try:
        return t, rekomendasi_strategi_gabungan(t)
    except Exception:
        return t, None


def rekomendasi_top_per_strategi(tickers: list, top_n: int = 3,
                                 max_workers: int = 6) -> dict:
    """
    Untuk tiap strategi: 3 saham terbaik versi skor + backtest historis finalis.

    Return: {
        'jumlah_saham_diproses': n,
        'jumlah_saham_berhasil': m,
        'saham_gagal': [...],
        'leaderboard_per_strategi': [
            {'kode': ..., 'nama': ..., 'terbaik': [
                {'ticker', 'skor', 'kecocokan', 'sinyal_aktif', 'harga',
                 'backtest': {metrik per hari, win rate, sample, andal} | None}
            ]},
            ...
        ]
    }
    """
    gabungan_by_saham = {}
    gagal = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_proses_satu_saham, t) for t in tickers]
        for fut in as_completed(futures):
            t, r = fut.result()
            if r:
                gabungan_by_saham[t] = r
            else:
                gagal.append(t)

    # --- Leaderboard per strategi (ranking skor) ---
    skor_by = {kode: [] for kode in NAMA_STRATEGI}
    for t, g in gabungan_by_saham.items():
        for p in g['peringkat_strategi']:
            kode = p['kode']
            if kode in skor_by:
                skor_by[kode].append({
                    'ticker': t,
                    'skor': p['skor'],
                    'kecocokan': p['kecocokan'],
                    'sinyal_aktif': p['sinyal_aktif'],
                    'harga': g['harga_saat_ini'],
                })

    leaderboard = []
    for kode, lst in skor_by.items():
        lst.sort(key=lambda x: -x['skor'])
        finalis = lst[:top_n]

        # --- Backtest historis tiap finalis (paralel antar finalis) ---
        def proses_finalis(f_):
            try:
                bt = STRATEGI_BACKTEST[kode]['fungsi'](f_['ticker'])
                return f_, _metrik_per_hari(kode, bt)
            except Exception:
                return f_, None

        with ThreadPoolExecutor(max_workers=top_n) as ex:
            futures = [ex.submit(proses_finalis, f_) for f_ in finalis]
            for fut in as_completed(futures):
                f_, met = fut.result()
                f_['backtest'] = met

        leaderboard.append({
            'kode': kode,
            'nama': NAMA_STRATEGI[kode],
            'terbaik': finalis,
        })

    return {
        'jumlah_saham_diproses': len(tickers),
        'jumlah_saham_berhasil': len(gabungan_by_saham),
        'saham_gagal': gagal,
        'leaderboard_per_strategi': leaderboard,
    }


def _ringkas_untuk_print(laporan: dict) -> str:
    baris = []
    baris.append(f"Diproses {laporan['jumlah_saham_diproses']} saham, "
                 f"{laporan['jumlah_saham_berhasil']} berhasil dianalisis.")
    if laporan['saham_gagal']:
        baris.append(f"Gagal: {', '.join(laporan['saham_gagal'][:8])}"
                     + (" ..." if len(laporan['saham_gagal']) > 8 else ""))
    for lb in laporan['leaderboard_per_strategi']:
        tipe = STRATEGI_BACKTEST[lb['kode']]['tipe']
        baris.append(f"\n=== {lb['nama']} ({lb['kode']}) ===")
        if not lb['terbaik']:
            baris.append("  (tidak ada saham dengan skor untuk strategi ini)")
        for i, f_ in enumerate(lb['terbaik'], 1):
            bt = f_['backtest']
            if bt:
                # Label metrik beda antara tipe per_transaksi vs per_hari — jangan
                # sampai 'ekspektasi per hari' atau 'fill rate' salah dicap 'winrate'.
                wr = bt.get('win_rate_persen')
                wr_txt = f"winrate {wr}%" if (tipe == 'per_transaksi' and wr is not None) else (
                    f"hari roundtrip terisi {wr}%" if (tipe == 'per_hari' and wr is not None) else "winrate -")  
                bt_txt = (
                    f"{wr_txt} | {bt['satuan_metrik_asli']}: {bt['metrik_asli_persen']}% "
                    f"({bt['metrik_per_hari_persen']}%/hari) | "
                    f"sample {bt['sample']} | "
                    f"{'ANDAL' if bt['andal'] else 'sampel kecil'}"
                )
            else:
                bt_txt = "backtest tidak tersedia (sampel 0)"
            baris.append(
                f"  {i}. {f_['ticker']}: skor {f_['skor']} | {f_['kecocokan']} | "
                f"sinyal: {f_['sinyal_aktif']} | harga Rp{f_['harga']} | {bt_txt}"
            )
    return "\n".join(baris)


if __name__ == "__main__":
    industri = "Bank"
    top_n = 3
    max_workers = 6
    for a in sys.argv[1:]:
        if a.startswith("--industri="):
            industri = a.split("=", 1)[1]
        elif a.startswith("--top="):
            try:
                top_n = int(a.split("=", 1)[1])
            except ValueError:
                print(f"Argumen tidak valid: {a}", file=sys.stderr)
        elif a.startswith("--max-workers="):
            try:
                max_workers = int(a.split("=", 1)[1])
            except ValueError:
                print(f"Argumen tidak valid: {a}", file=sys.stderr)

    print(f"Mencari saham dengan industry mengandung '{industri}'...", flush=True)
    tickers = daftar_saham_sektor(industri)
    if not tickers:
        print(f"⚠️ Tidak ada saham dengan industry mengandung '{industri}'. "
              f"Coba kata kunci lain (misal: Bank, Coal, Real Estate, Technology).", file=sys.stderr)
        sys.exit(1)
    print(f"Ditemukan {len(tickers)} saham. Menghitung skor strategi gabungan "
          f"(max_workers={max_workers})...", flush=True)

    laporan = rekomendasi_top_per_strategi(tickers, top_n=top_n, max_workers=max_workers)
    file_out = f"rekomendasi_sektor_{industri.lower().replace(' ', '_')}.json"
    with open(file_out, "w") as f:
        json.dump(laporan, f, ensure_ascii=False, indent=2, default=str)
    print(f"Laporan tersimpan: {file_out}\n")
    print(_ringkas_untuk_print(laporan))
