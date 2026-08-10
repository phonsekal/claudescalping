# app/config.py
TARGET_DIVIDEND_YIELD = 0.07
PE_WAJAR_BANK = 12.0
PE_WAJAR_UMUM = 15.0

# Ambang minimal yield dividen supaya dianggap layak jadi BASIS VALUASI utama.
# Saham yang secara teknis bagi dividen tapi nominalnya receh (di bawah ambang ini)
# diperlakukan seperti saham non-dividen untuk keperluan valuasi (pakai harga wajar
# PE-based sebagai acuan), bukan rumus dividend-yield yang bisa menghasilkan angka
# tidak masuk akal untuk yield yang sangat kecil.
YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN = 0.01  # 1%

# Watchlist awal untuk strategi Swing-Investment Dividen (Big & Medium Caps)
INDEX_BLUECHIP_UTAMA = ["BBRI.JK", "BMRI.JK", "BBNI.JK", "BBCA.JK", "ASII.JK", "TLKM.JK", "UNVR.JK", "PTBA.JK", "ADRO.JK", "ANTM.JK", "ICBP.JK", "INDF.JK", "AMRT.JK", "SIDO.JK"]

# Watchlist awal untuk strategi Saham Gorengan Spekulatif (High Volatility / Penny Stocks)
WATCHLIST_GORENGAN = ["JGLE.JK", "BUMI.JK", "BRMS.JK", "DEWA.JK", "DOOH.JK", "GOTO.JK", "WIFI.JK", "JKON.JK"]

# --- KONFIGURASI TAMBAHAN (STRATEGI YANG DIMAKSIMALKAN) ---

# Ticker indeks acuan untuk filter kondisi makro market sebelum sinyal individual dipakai
MARKET_INDEX_TICKER = "^JKSE"

# Ambang ADX untuk strategi gorengan (20 = default longgar, 25 = lebih ketat/selektif)
ADX_THRESHOLD_GORENGAN = 20.0

# Ambang klasifikasi kualitas tren ADX (standar interpretasi Wilder):
# < 20 = tidak ada tren, 20-25 = tren baru terbentuk (moderat), 25-40 = tren kuat,
# > 40 = tren sangat kuat/ekstrem (waspada jenuh)
ADX_TREN_MODERAT = 20.0
ADX_TREN_KUAT = 25.0
ADX_TREN_EKSTREM = 40.0

# --- KONFIGURASI REKOMENDASI HARGA ENTRY DAYTRADING ---
# Estimasi total biaya transaksi bolak-balik (fee beli ~0.15% + fee jual ~0.25%,
# angka umum broker online Indonesia; sesuaikan dengan fee broker kamu)
FEE_TRANSAKSI_TOTAL_PERSEN = 0.40
# Profit bersih minimal (setelah fee) yang masih dianggap layak untuk daytrading.
# Dipakai untuk menghitung "harga masuk maksimal yang masih memberikan profit":
# harga tertinggi yang, jika target jual tercapai, masih menyisakan profit bersih
# minimal sebesar angka ini.
MIN_PROFIT_BERSIH_DAYTRADING_PERSEN = 2.0

# Rasio minimal potensi profit dibanding potensi rugi (reward:risk) supaya sebuah
# setup daytrading dianggap layak. Di bawah ini, setup ditolak walau tren ADX kuat —
# contoh kasus: KBLV Agu 2026, tren kuat tapi jarak ke resisten hampir sama dengan
# jarak ke stop loss (rasio ~1:1) = ekspektasi matematis buruk.
MIN_RASIO_RISK_REWARD_DAYTRADING = 1.5

# Batas RSI maksimal untuk masih memberi rekomendasi entry daytrading. Di atas ini
# harga dianggap parabolik/jenuh beli ekstrem (contoh kasus: CBPE dengan RSI 90) —
# masuk di kondisi ini artinya membeli di pucuk euforia.
RSI_MAKS_UNTUK_ENTRY_DAYTRADING = 80.0

# Kelipatan ATR untuk menghitung TP/SL gorengan secara proporsional ke volatilitas saham,
# menggantikan angka persentase flat (7% / 3.5%) yang sama rata untuk semua saham.
# TP dinaikkan 3.0 -> 4.0 berdasarkan backtest gabungan watchlist gorengan (Agu 2026):
# rata-rata return per transaksi naik ~+2.3% -> ~+3.2% dengan win rate sama (50%).
# TP 5.0 tambahannya sudah tipis; trailing stop justru memperburuk hasil.
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 4.0

# Alokasi bertingkat (persen) untuk 3 tranche average-down di strategi dividen.
# Alokasi makin besar di level koreksi yang makin dalam.
TRANCHE_ALLOKASI = [20, 30, 50]

# Batas toleransi penurunan dividen tahun berjalan vs tahun lalu sebelum guardrail
# fundamental menyatakan "tidak aman untuk average down" (circuit breaker pengganti cut loss)
BATAS_TOLERANSI_PENURUNAN_DIVIDEN = 0.30  # 30%

# --- KONFIGURASI STRATEGI 3: FAST INTRADAY ALERT ---
# CATATAN JUJUR (dibahas dengan user Agu 2026): ini BUKAN scalping asli. Scalping
# sungguhan butuh data tick/real-time dan eksekusi otomatis dalam hitungan detik —
# Yahoo Finance delay ~15-20 menit untuk saham IDX, dan alur kita (WA -> n8n ->
# Vercel -> Yahoo -> balik ke WA) makan beberapa detik. Ini alat SINYAL untuk
# trader manual dengan horizon lebih pendek dari gorengan (puluhan menit - ~2 jam),
# bukan otomasi scalping bermilidetik.
INTERVAL_FAST_INTRADAY = "15m"
PERIODE_DATA_FAST_INTRADAY = "5d"  # cukup untuk sinyal terkini, fetch lebih ringan/cepat

# Watchlist fast-intraday: saham LIKUID (spread tipis), BUKAN saham gorengan volatile.
# Target profit tipis (1.5-2%) butuh spread kecil, kalau tidak keburu habis dimakan
# spread+fee sebelum sempat profit — beda filosofi dari gorengan yang justru cari
# volatilitas tinggi.
WATCHLIST_FAST_INTRADAY = [
    "ACES.JK", "ADRO.JK", "AKRA.JK", "AMMN.JK", "AMRT.JK", 
    "ANTM.JK", "ARTO.JK", "ASII.JK", "BBCA.JK", "BBNI.JK", 
    "BBRI.JK", "BBTN.JK", "BMRI.JK", "BRIS.JK", "BRPT.JK", 
    "BUKA.JK", "CPIN.JK", "EMTK.JK", "ESSA.JK", "EXCL.JK", 
    "GGRM.JK", "GOTO.JK", "HRUM.JK", "ICBP.JK", "INDF.JK", 
    "INKP.JK", "INCO.JK", "INTP.JK", "ITMG.JK", "KLBF.JK", 
    "MAPI.JK", "MBMA.JK", "MDKA.JK", "MEDC.JK", "MTEL.JK", 
    "PGAS.JK", "PGEO.JK", "PTBA.JK", "SIDO.JK", "SMGR.JK", 
    "SRTG.JK", "TLKM.JK", "TOWR.JK", "UNTR.JK", "UNVR.JK"
]
MAX_HOLD_BARS_FAST_INTRADAY = 8  # 8 x 15 menit = maks ~2 jam hold yang disarankan
MIN_PROFIT_BERSIH_FAST_INTRADAY_PERSEN = 1.5
MIN_RASIO_RISK_REWARD_FAST_INTRADAY = 1.5
ATR_MULTIPLIER_SL_FAST_INTRADAY = 1.2
ATR_MULTIPLIER_TP_FAST_INTRADAY = 2.0
VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY = 2.0
JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY = 20

# --- KONFIGURASI RETRY (ketahanan terhadap kegagalan sesaat Yahoo Finance) ---
RETRY_PERCOBAAN_MAKSIMAL = 2
RETRY_JEDA_DETIK = 1.5
# TTL cache untuk kondisi market IHSG — menghindari download ulang data ^JKSE yang
# sama saat banyak request masuk dalam waktu berdekatan (misal screener 50+ saham
# yang masing-masing memanggil cek_kondisi_market). Kondisi IHSG jarang berubah
# dalam hitungan menit, jadi cache 5 menit aman.
CACHE_TTL_MARKET_DETIK = 300

# --- KONFIGURASI MANAJEMEN RISIKO PORTOFOLIO (saran ukuran posisi) ---
# CATATAN: API ini stateless (tidak ada database), jadi angka di bawah ini adalah
# SARAN batas per posisi, bukan pelacakan otomatis posisi yang sudah kamu buka.
# Kamu tetap perlu mencatat sendiri total alokasi aktifmu di luar sistem ini,
# kecuali nanti ditambahkan lapisan penyimpanan (database) terpisah.
MAX_ALOKASI_SWING_PERSEN = 15          # maks % modal per saham dividen/swing
MAX_ALOKASI_GORENGAN_PERSEN = 5        # maks % modal per saham gorengan (lebih kecil, risiko tinggi)
MAX_JUMLAH_GORENGAN_BERSAMAAN = 3      # saran jumlah maksimal posisi gorengan dibuka bersamaan

# --- UNIVERSE SAHAM UNTUK SCREENER (LEGACY) ---
# CATATAN: endpoint /screener/*/semua-saham kini memakai daftar LENGKAP 941 saham
# tercatat BEI di app/daftar_saham_bei.py, bukan lagi starter list ini.
# List di bawah dipertahankan hanya untuk kompatibilitas / bahan watchlist manual.
SEMUA_SAHAM_IDX_STARTER = [
    # Perbankan
    "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "ARTO.JK", "BJBR.JK", "BJTM.JK", "BTPS.JK", "BNGA.JK",
    # Consumer / Ritel
    "UNVR.JK", "ICBP.JK", "INDF.JK", "MYOR.JK", "ULTJ.JK", "SIDO.JK", "AMRT.JK", "MIDI.JK", "MAPI.JK", "ACES.JK",
    "ERAA.JK", "LPPF.JK", "RALS.JK", "CPIN.JK", "JPFA.JK", "GGRM.JK", "HMSP.JK", "KLBF.JK", "KAEF.JK", "TSPC.JK",
    # Telco / Infrastruktur Digital
    "TLKM.JK", "EXCL.JK", "ISAT.JK", "TOWR.JK", "TBIG.JK", "GOTO.JK", "BUKA.JK", "EMTK.JK", "WIFI.JK",
    # Otomotif / Manufaktur
    "ASII.JK", "UNTR.JK", "AUTO.JK", "SMSM.JK",
    # Pertambangan / Energi
    "PTBA.JK", "ADRO.JK", "ITMG.JK", "INDY.JK", "HRUM.JK", "ANTM.JK", "INCO.JK", "MDKA.JK", "TINS.JK", "BUMI.JK",
    "BRMS.JK", "MEDC.JK", "ELSA.JK", "PGAS.JK", "AKRA.JK", "BSSR.JK", "DSSA.JK", "PSAB.JK", "MBMA.JK", "AMMN.JK",
    "NCKL.JK",
    # Semen / Properti / Konstruksi
    "SMGR.JK", "INTP.JK", "WSKT.JK", "WIKA.JK", "PTPP.JK", "JSMR.JK", "PWON.JK", "BSDE.JK", "CTRA.JK", "SMRA.JK",
    "LPKR.JK", "ASRI.JK",
    # Perkebunan
    "AALI.JK", "LSIP.JK", "SMAR.JK", "TAPG.JK",
    # Media / Teknologi
    "SCMA.JK", "MNCN.JK",
    # Kimia / Petrokimia
    "BRPT.JK", "TPIA.JK",
    # Saham Gorengan / High Volatility
    "DEWA.JK", "DOOH.JK", "JGLE.JK", "JKON.JK", "BULL.JK", "RAJA.JK", "PANI.JK", "BREN.JK", "CUAN.JK",
]
