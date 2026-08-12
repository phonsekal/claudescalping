# app/validasi_strategi.py
"""
Validasi skor kecocokan strategi vs performa backtest historis.

Tujuan: mengecek apakah strategi yang dinilai "paling cocok" oleh
rekomendasi_strategi_gabungan (skor kecocokan) benar-benar yang paling bagus
performa historisnya menurut backtest masing-masing strategi.

MASALAH SATUAN & CARA MENGATASINYA:
- Metrik antar strategi tidak sama satuan:
  * swing / gorengan / fast-intraday / BSJP → 'rata_rata_return_per_transaksi'
    (backtest mensimulasikan transaksi satu per satu).
  * range pagi-sore / BPJS → 'ekspektasi_per_hari' (backtest walk-forward
    menghitung ekspektasi bersih per hari bursa).
- Keduanya dinormalisasi ke "perkiraan ekspektasi per hari" dengan asumsi yang
  DIDOKUMENTASIKAN di tiap strategi (misal: swing diasumsikan 1 transaksi
  ditahan ~21 hari bursa). Normalisasi ini kasar — jangan dibaca sebagai angka
  profit yang presisi, tapi sebagai urutan/perbandingan yang adil.

KETERBATASAN YANG HARUS DISADARI:
- Backtest intraday (gorengan, fast-intraday, range pagi-sore, BPJS) cuma punya
  sampel data ~60 hari (batas yfinance), jadi sampelnya kecil dan hasilnya
  TIDAK statistically robust. Flag 'andal' menandai strategi dengan sampel
  cukup untuk dipercaya.
- 'cocok_top1' = strategi terbaik versi skor == strategi terbaik versi backtest.
  Kecocokan penuh tidak diharapkan di semua saham — yang dicari adalah apakah
  arah umum skor sejalan dengan performa historis (korelasi Spearman + % cocok).
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services import rekomendasi_strategi_gabungan
from app.backtest import (
    backtest_swing_dividen, backtest_gorengan_momentum, backtest_fast_intraday,
    backtest_bsjp, backtest_range_pagi_sore, backtest_bpjs,
)

NAMA_STRATEGI = {
    'swing': 'Swing-Investment Dividen',
    'gorengan': 'Momentum Gorengan (Day Trading)',
    'fast_intraday': 'Fast Intraday Alert (15 menit)',
    'bsjp': 'BSJP (Beli Sore Jual Pagi)',
    'range_pagi_sore': 'Range Pagi-Sore (Jual Pagi, Beli Sore)',
    'bpjs': 'BPJS (Beli Pagi Jual Sore)',
}

# Pemetaan kode strategi -> fungsi backtest + konfigurasi normalisasi metrik.
# divider_hari: asumsi rata-rata 1 transaksi ditahan berapa hari bursa, dipakai
# untuk mengubah 'return per transaksi' menjadi 'perkiraan return per hari'.
STRATEGI_BACKTEST = {
    'swing': {
        'fungsi': backtest_swing_dividen,
        'tipe': 'per_transaksi',
        'divider_hari': 21,      # swing: hold ~3-4 minggu bursa (backtest max 30 hari)
        'min_sample': 10,        # bluechip stabil jarang memicu >10 sinyal oversold dalam 2 tahun
        'asumsi': "return per transaksi / 21 hari bursa (asumsi hold ~1 bulan)",
    },
    'gorengan': {
        'fungsi': backtest_gorengan_momentum,
        'tipe': 'per_transaksi',
        'divider_hari': 1,       # gorengan: 1 transaksi ≈ 1 hari bursa
        'min_sample': 5,
        'asumsi': "return per transaksi (1 transaksi ≈ 1 hari bursa)",
    },
    'fast_intraday': {
        'fungsi': backtest_fast_intraday,
        'tipe': 'per_transaksi',
        'divider_hari': 1,       # fast-intraday: 1 transaksi ≈ 1 hari bursa
        'min_sample': 5,
        'asumsi': "return per transaksi (1 transaksi ≈ 1 hari bursa)",
    },
    'bsjp': {
        'fungsi': backtest_bsjp,
        'tipe': 'per_transaksi',
        'divider_hari': 1,       # BSJP: beli sore, jual pagi besok = 1 hari
        'min_sample': 10,        # frekuensi sinyal BSJP di large-cap memang rendah (lihat config.py)
        'asumsi': "return per transaksi (1 posisi semalam ≈ 1 hari bursa)",
    },
    'range_pagi_sore': {
        'fungsi': backtest_range_pagi_sore,
        'tipe': 'per_hari',
        'min_sample': 20,
        'asumsi': "ekspektasi bersih per hari dari backtest walk-forward (sudah per hari)",
    },
    'bpjs': {
        'fungsi': backtest_bpjs,
        'tipe': 'per_hari',
        'min_sample': 20,
        'asumsi': "ekspektasi bersih per hari dari backtest walk-forward (sudah per hari)",
    },
}


def _rata_hari_hold_dari_detail(kode: str, hasil: dict) -> float or None:
    """
    Estimasi rata-rata lama 1 transaksi di-hold (dalam HARI bursa) dari detail
    transaksi yang dikembalikan backtest. Dipakai untuk normalisasi 'return per
    transaksi' -> 'return per hari' yang lebih jujur daripada asumsi flat.

    - swing: field 'hari_ditahan' (sudah dalam hari bursa).
    - gorengan: field 'jam_ditahan' -> / 6.5 jam per sesi IDX.
    - fast-intraday: field 'bar_15menit_ditahan' -> x 15 menit / (6.5 jam x 60).
    - BSJP: posisi semalam -> 1 hari bursa (fixed).

    Return None jika detail tidak tersedia (fallback ke asumsi flat di config).
    """
    if kode == 'bsjp':
        return 1.0
    detail = hasil.get('detail_transaksi_terakhir') or []
    if not detail:
        return None

    if kode == 'swing':
        nilai = [d.get('hari_ditahan') for d in detail if d.get('hari_ditahan') is not None]
        if not nilai:
            return None
        return round(sum(float(v) for v in nilai) / len(nilai), 2)
    if kode == 'gorengan':
        nilai = [d.get('jam_ditahan') for d in detail if d.get('jam_ditahan') is not None]
        if not nilai:
            return None
        return round(sum(float(v) for v in nilai) / len(nilai) / 6.5, 3)
    if kode == 'fast_intraday':
        nilai = [d.get('bar_15menit_ditahan') for d in detail if d.get('bar_15menit_ditahan') is not None]
        if not nilai:
            return None
        return round(sum(float(v) for v in nilai) / len(nilai) * 15 / (6.5 * 60), 3)
    return None


def _metrik_per_hari(kode: str, hasil: dict) -> dict or None:
    """
    Ekstrak & normalisasi metrik backtest satu strategi ke 'perkiraan per hari'.

    Normalisasi memakai RATA-RATA LAMA HOLD REAL dari detail transaksi bila
    tersedia (lebih jujur daripada asumsi flat), dan fallback ke asumsi di
    STRATEGI_BACKTEST kalau detail tidak ada.

    Return None jika backtest gagal / tidak ada data cukup (strategi ini tidak
    masuk peringkat backtest).
    """
    if not hasil:
        return None
    cfg = STRATEGI_BACKTEST[kode]

    if cfg['tipe'] == 'per_transaksi':
        total = hasil.get('total_transaksi', 0)
        avg = hasil.get('rata_rata_return_per_transaksi_persen')
        if total == 0 or avg is None:
            return None
        avg = float(avg)

        hari_hold = _rata_hari_hold_dari_detail(kode, hasil)
        if hari_hold and hari_hold > 0:
            divider = hari_hold
            asumsi = f"return per transaksi / {divider} hari bursa (rata-rata hold real dari detail backtest)"
        else:
            divider = cfg['divider_hari']
            asumsi = cfg['asumsi']

        return {
            'kode': kode,
            'nama': NAMA_STRATEGI[kode],
            'metrik_asli_persen': round(avg, 2),
            'metrik_per_hari_persen': round(avg / divider, 3),
            'satuan_metrik_asli': 'rata-rata return per transaksi (%)',
            'asumsi_per_hari': asumsi,
            'sample': int(total),
            'andal': int(total) >= cfg['min_sample'],
            'win_rate_persen': hasil.get('win_rate_persen'),
            'max_drawdown_persen': hasil.get('max_drawdown_persen'),
            'catatan_backtest': hasil.get('catatan', ''),
        }

    # tipe per_hari (range pagi-sore & BPJS)
    hari = hasil.get('jumlah_hari_diuji', 0)
    eksp = hasil.get('ekspektasi_per_hari_persen')
    if hari == 0 or eksp is None:
        return None
    return {
        'kode': kode,
        'nama': NAMA_STRATEGI[kode],
        'metrik_asli_persen': round(float(eksp), 2),
        'metrik_per_hari_persen': round(float(eksp), 3),
        'satuan_metrik_asli': 'ekspektasi bersih per hari (walk-forward)',
        'asumsi_per_hari': cfg['asumsi'],
        'sample': int(hari),
        'andal': int(hari) >= cfg['min_sample'],
        'win_rate_persen': hasil.get('persen_hari_roundtrip_terisi'),
        'max_drawdown_persen': None,
        'catatan_backtest': hasil.get('catatan', ''),
    }


def _spearman(a: list, b: list) -> float or None:
    """
    Korelasi peringkat Spearman tanpa scipy (rank average untuk nilai kembar).
    None jika < 3 data atau varians nol (tidak bisa dihitung).
    """
    n = len(a)
    if n < 3 or len(a) != len(b):
        return None

    def _rank(xs):
        urut = sorted(range(len(xs)), key=lambda i: xs[i])
        rank = [0.0] * len(xs)
        i = 0
        while i < len(urut):
            j = i
            while j + 1 < len(urut) and xs[urut[j + 1]] == xs[urut[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rank[urut[k]] = avg
            i = j + 1
        return rank

    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return None
    return round(cov / (va * vb) ** 0.5, 2)


def validasi_kecocokan_satu_saham(ticker_symbol: str, kondisi_market: dict = None) -> dict or None:
    """
    Bandingkan peringkat skor kecocokan vs peringkat performa backtest untuk 1 saham.

    Langkah:
    1. rekomendasi_strategi_gabungan -> peringkat skor (6 strategi).
    2. Jalankan SEMUA backtest strategi (paralel) -> peringkat performa per hari.
    3. Hitung: cocok_top1 (terbaik skor == terbaik backtest) & korelasi Spearman
       antara skor dan metrik per hari untuk strategi yang punya keduanya.
    """
    gabungan = rekomendasi_strategi_gabungan(ticker_symbol, kondisi_market=kondisi_market)
    if not gabungan:
        return None

    peringkat_skor = gabungan['peringkat_strategi']
    skor_by_kode = {p['kode']: p['skor'] for p in peringkat_skor}

    # Jalankan semua backtest secara paralel (tiap backtest menarik data sendiri).
    hasil_bt = {}

    def proses(kode):
        try:
            return kode, STRATEGI_BACKTEST[kode]['fungsi'](ticker_symbol)
        except Exception:
            return kode, None

    with ThreadPoolExecutor(max_workers=len(STRATEGI_BACKTEST)) as executor:
        futures = [executor.submit(proses, k) for k in STRATEGI_BACKTEST]
        for future in as_completed(futures):
            kode, res = future.result()
            hasil_bt[kode] = _metrik_per_hari(kode, res)

    peringkat_bt = sorted(
        [h for h in hasil_bt.values() if h is not None],
        key=lambda h: h['metrik_per_hari_persen'],
        reverse=True,
    )
    bt_by_kode = {h['kode']: h for h in peringkat_bt}

    terbaik_skor = peringkat_skor[0]['kode'] if peringkat_skor else None
    terbaik_bt = peringkat_bt[0]['kode'] if peringkat_bt else None
    cocok_top1 = bool(terbaik_skor and terbaik_bt and terbaik_skor == terbaik_bt)

    # Korelasi Spearman antara skor & metrik per hari (strategi yang punya keduanya)
    pasangan = [(skor_by_kode[k], h['metrik_per_hari_persen']) for k, h in bt_by_kode.items() if k in skor_by_kode]
    korelasi = _spearman([x[0] for x in pasangan], [x[1] for x in pasangan]) if len(pasangan) >= 3 else None

    # --- METRIK VALIDASI TAMBAHAN (lebih tahan terhadap sampel kecil) ---
    # 1) cocok_top1_andal: top-1 hanya di antara strategi yang sampel backtestnya andal.
    # 2) tumpang_tindih_top3: berapa dari 3 strategi skor teratas juga masuk 3 besar backtest.
    # 3) peringkat_skor_terbaik_backtest: di urutan berapa (1=satu) strategi pemenang backtest
    #    menurut skor — kalau pemenang backtest ternyata di skor terbawah, itu sinyal
    #    peringkat skor dan performa historis benar-benar berlawanan.
    kode_top3_skor = [p['kode'] for p in peringkat_skor[:3]]
    kode_top3_bt = [h['kode'] for h in peringkat_bt[:3]]
    tumpang_tindih_top3 = len(set(kode_top3_skor) & set(kode_top3_bt))

    bt_andal = [h for h in peringkat_bt if h['andal']]
    terbaik_bt_andal = bt_andal[0]['kode'] if bt_andal else None
    cocok_top1_andal = bool(terbaik_skor and terbaik_bt_andal and terbaik_skor == terbaik_bt_andal)

    peringkat_skor_terbaik_backtest = None
    if terbaik_bt:
        for i, p in enumerate(peringkat_skor):
            if p['kode'] == terbaik_bt:
                peringkat_skor_terbaik_backtest = i + 1
                break

    # Korelasi Spearman khusus strategi dengan sampel andal
    pasangan_andal = [
        (skor_by_kode[k], h['metrik_per_hari_persen'])
        for k, h in bt_by_kode.items()
        if k in skor_by_kode and h['andal']
    ]
    korelasi_andal = _spearman(
        [x[0] for x in pasangan_andal], [x[1] for x in pasangan_andal]
    ) if len(pasangan_andal) >= 3 else None

    strategi_tanpa_bt = [k for k in STRATEGI_BACKTEST if hasil_bt.get(k) is None]

    catatan = []
    if terbaik_skor and terbaik_skor not in bt_by_kode:
        catatan.append(
            f"Strategi terbaik versi skor ({NAMA_STRATEGI[terbaik_skor]}) tidak punya data backtest "
            "cukup — tidak bisa dibandingkan performa historisnya."
        )
    bt_top = bt_by_kode.get(terbaik_skor) if terbaik_skor else None
    if bt_top and not bt_top['andal']:
        catatan.append(
            f"Strategi terbaik versi skor ({NAMA_STRATEGI[terbaik_skor]}) sampel backtest kecil "
            f"({bt_top['sample']}) — performa historisnya belum andal secara statistik."
        )
    if peringkat_skor_terbaik_backtest and peringkat_skor_terbaik_backtest >= 4:
        catatan.append(
            f"⚠️ Pemenang backtest ({NAMA_STRATEGI[terbaik_bt]}) hanya berada di peringkat skor "
            f"ke-{peringkat_skor_terbaik_backtest} — skor dan performa historis cenderung bertolak belakang."
        )
    if not catatan:
        catatan.append("Semua strategi dengan skor punya data backtest yang cukup untuk dibandingkan.")

    return {
        "saham": gabungan['saham'],
        "harga_saat_ini": gabungan['harga_saat_ini'],
        "kondisi_market": gabungan.get('kondisi_market'),
        "profil_saham": gabungan.get('profil_saham'),
        "strategi_terbaik_versi_skor": terbaik_skor,
        "strategi_terbaik_versi_backtest": terbaik_bt,
        "cocok_top1": cocok_top1,
        "cocok_top1_hanya_sampel_andal": cocok_top1_andal,
        "tumpang_tindih_top3": tumpang_tindih_top3,
        "peringkat_skor_terbaik_backtest": peringkat_skor_terbaik_backtest,
        "korelasi_spearman_skor_vs_backtest": korelasi,
        "korelasi_spearman_hanya_sampel_andal": korelasi_andal,
        "peringkat_skor": [
            {
                "kode": p['kode'],
                "nama": p['nama'],
                "skor": p['skor'],
                "kecocokan": p['kecocokan'],
                "sinyal_aktif": p['sinyal_aktif'],
            }
            for p in peringkat_skor
        ],
        "peringkat_backtest": [
            {
                "kode": h['kode'],
                "nama": h['nama'],
                "metrik_asli_persen": h['metrik_asli_persen'],
                "metrik_per_hari_persen": h['metrik_per_hari_persen'],
                "satuan_metrik_asli": h['satuan_metrik_asli'],
                "asumsi_per_hari": h['asumsi_per_hari'],
                "sample": h['sample'],
                "andal": h['andal'],
                "win_rate_persen": h['win_rate_persen'],
                "max_drawdown_persen": h['max_drawdown_persen'],
            }
            for h in peringkat_bt
        ],
        "strategi_tanpa_data_backtest": [NAMA_STRATEGI[k] for k in strategi_tanpa_bt],
        "catatan": catatan,
    }


def validasi_kecocokan_watchlist(tickers: list, max_workers: int = 5, kondisi_market: dict = None) -> dict:
    """
    Validasi lintas BANYAK saham sekaligus (paralel), lalu agregasi:
    - % saham di mana terbaik-skor == terbaik-backtest (cocok_top1)
    - rata-rata korelasi Spearman skor vs backtest
    - ringkasan per strategi: berapa kali jadi terbaik versi skor vs backtest
      + rata-rata metrik per hari saat jadi terbaik-skor
    """
    hasil = {}

    def proses(t):
        try:
            return t, validasi_kecocokan_satu_saham(t, kondisi_market=kondisi_market)
        except Exception:
            return t, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in tickers]
        for future in as_completed(futures):
            t, r = future.result()
            hasil[t] = r

    valid = {t: r for t, r in hasil.items() if r}
    n = len(valid)
    if n == 0:
        return {"jumlah_saham_diuji": 0, "keterangan": "Tidak ada saham yang berhasil divalidasi."}

    cocok = [r['cocok_top1'] for r in valid.values()]
    cocok_andal = [
        r['cocok_top1_hanya_sampel_andal']
        for r in valid.values()
        if r.get('cocok_top1_hanya_sampel_andal') is not None
    ]
    top3_overlap = [r['tumpang_tindih_top3'] for r in valid.values()]
    korelasi_list = [
        r['korelasi_spearman_skor_vs_backtest']
        for r in valid.values()
        if r.get('korelasi_spearman_skor_vs_backtest') is not None
    ]
    korelasi_andal_list = [
        r['korelasi_spearman_hanya_sampel_andal']
        for r in valid.values()
        if r.get('korelasi_spearman_hanya_sampel_andal') is not None
    ]

    # Ringkasan per strategi
    ringkas = {}
    for kode, nama in NAMA_STRATEGI.items():
        jadi_skor = sum(1 for r in valid.values() if r['strategi_terbaik_versi_skor'] == kode)
        jadi_bt = sum(1 for r in valid.values() if r['strategi_terbaik_versi_backtest'] == kode)
        metrik_saat_jadi_skor = [
            h['metrik_per_hari_persen']
            for r in valid.values()
            for h in r['peringkat_backtest']
            if h['kode'] == kode and r['strategi_terbaik_versi_skor'] == kode
        ]
        ringkas[kode] = {
            "nama": nama,
            "terbaik_versi_skor": jadi_skor,
            "terbaik_versi_backtest": jadi_bt,
            "rata2_metrik_per_hari_saat_jadi_terbaik_skor": (
                round(sum(metrik_saat_jadi_skor) / len(metrik_saat_jadi_skor), 3)
                if metrik_saat_jadi_skor else None
            ),
        }

    persen_cocok = round(sum(cocok) / n * 100, 1)
    persen_cocok_andal = round(sum(cocok_andal) / len(cocok_andal) * 100, 1) if cocok_andal else None
    rata_overlap = round(sum(top3_overlap) / len(top3_overlap), 2) if top3_overlap else None
    rata_korelasi = round(sum(korelasi_list) / len(korelasi_list), 2) if korelasi_list else None
    rata_korelasi_andal = round(sum(korelasi_andal_list) / len(korelasi_andal_list), 2) if korelasi_andal_list else None

    return {
        "jumlah_saham_diuji": n,
        "persen_cocok_top1_persen": persen_cocok,
        "persen_cocok_top1_hanya_sampel_andal_persen": persen_cocok_andal,
        "rata_rata_tumpang_tindih_top3": rata_overlap,
        "rata_rata_korelasi_spearman": rata_korelasi,
        "rata_rata_korelasi_spearman_hanya_sampel_andal": rata_korelasi_andal,
        "detail_per_saham": [
            {
                "saham": r['saham'],
                "terbaik_skor": r['strategi_terbaik_versi_skor'],
                "terbaik_backtest": r['strategi_terbaik_versi_backtest'],
                "cocok_top1": r['cocok_top1'],
                "cocok_top1_andal": r.get('cocok_top1_hanya_sampel_andal'),
                "tumpang_tindih_top3": r['tumpang_tindih_top3'],
                "peringkat_skor_terbaik_backtest": r['peringkat_skor_terbaik_backtest'],
                "korelasi_spearman": r['korelasi_spearman_skor_vs_backtest'],
                "korelasi_spearman_andal": r.get('korelasi_spearman_hanya_sampel_andal'),
            }
            for r in valid.values()
        ],
        "ringkasan_per_strategi": ringkas,
        "interpretasi": (
            f"Dari {n} saham yang diuji, {persen_cocok}% menunjukkan strategi terbaik versi skor == "
            f"strategi terbaik versi backtest. "
            + (f"Jika dibatasi strategi bersampel andal saja: {persen_cocok_andal}%. " if persen_cocok_andal is not None else "")
            + (f"Rata-rata 3 strategi skor teratas yang juga masuk 3 besar backtest: {rata_overlap} dari 3. " if rata_overlap is not None else "")
            + (f"Korelasi Spearman rata-rata {rata_korelasi} (semua strategi) dan {rata_korelasi_andal} "
               "(sampel andal saja; rentang -1 s.d. +1, > 0 berarti skor & performa historis searah)."
               if rata_korelasi is not None else "")
            + " Kecocokan < 100% adalah normal — skor juga mempertimbangkan sinyal saat ini & profil "
              "risiko, bukan hanya performa historis. Namun jika korelasi mendekati 0/negatif secara "
              "konsisten, skor kecocokan perlu dikalibrasi ulang."
        ),
        "peringatan": (
            "⚠️ Sampel backtest intraday (gorengan, fast-intraday, range pagi-sore, BPJS) terbatas "
            "~60 hari data, dan beberapa strategi hanya memicu beberapa transaksi di saham tertentu "
            "(misal swing 1-4 transaksi di bluechip stabil) — hasilnya sanity-check kasar, bukan bukti "
            "statistik. Perhatikan flag 'andal' per strategi sebelum menarik kesimpulan kuat."
        ),
    }
