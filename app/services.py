# app/services.py
import time
import yfinance as yf
import pandas as pd
import numpy as np
from app.config import (
    TARGET_DIVIDEND_YIELD, PE_WAJAR_BANK, PE_WAJAR_UMUM,
    MARKET_INDEX_TICKER, ADX_THRESHOLD_GORENGAN,
    ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP,
    TRANCHE_ALLOKASI, BATAS_TOLERANSI_PENURUNAN_DIVIDEN,
    RETRY_PERCOBAAN_MAKSIMAL, RETRY_JEDA_DETIK,
    MAX_ALOKASI_SWING_PERSEN, MAX_ALOKASI_GORENGAN_PERSEN, MAX_JUMLAH_GORENGAN_BERSAMAAN,
    YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN,
    ADX_TREN_MODERAT, ADX_TREN_KUAT, ADX_TREN_EKSTREM,
    FEE_TRANSAKSI_TOTAL_PERSEN, MIN_PROFIT_BERSIH_DAYTRADING_PERSEN,
    MIN_RASIO_RISK_REWARD_DAYTRADING, RSI_MAKS_UNTUK_ENTRY_DAYTRADING,
    MAX_HOLD_BARS_FAST_INTRADAY, MIN_PROFIT_BERSIH_FAST_INTRADAY_PERSEN,
    MIN_RASIO_RISK_REWARD_FAST_INTRADAY, ATR_MULTIPLIER_SL_FAST_INTRADAY,
    ATR_MULTIPLIER_TP_FAST_INTRADAY, VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY,
    JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY, INTERVAL_FAST_INTRADAY,
    PERIODE_DATA_FAST_INTRADAY,
    CACHE_TTL_MARKET_DETIK,
    BSJP_GAIN_MIN_PERSEN, BSJP_CLOSE_POSISI_RANGE_MIN, BSJP_VOLUME_MULTIPLIER,
    BSJP_VALUE_MIN_RUPIAH, BSJP_RSI_MAKS, BSJP_ADX_MIN, BSJP_SL_PAGI_PERSEN,
    BSJP_TARGET_PERSEN, BSJP_PERIODE_DATA,
    RANGE_PAGI_SORE_PERIODE, RANGE_PAGI_SORE_INTERVAL, RANGE_PAGI_SORE_JAM_PAGI,
    RANGE_PAGI_SORE_JAM_SORE, RANGE_PAGI_SORE_WINDOW_HARI,
    RANGE_PAGI_SORE_PERSENTIL_JUAL, RANGE_PAGI_SORE_PERSENTIL_BELI
)
import datetime

# Cache in-memory untuk kondisi market IHSG (TTL-based, menghindari download
# ulang data ^JKSE yang sama dalam waktu singkat saat banyak request bersamaan)
_cache_kondisi_market = {"data": None, "waktu": 0}


# =========================================================================
# HELPER: BATCH FETCH & RETRY (mempercepat & menstabilkan pengambilan data)
# =========================================================================

def ambil_riwayat_batch(tickers: list, period: str = "1y", interval: str = "1d"):
    """
    Tarik data historis BANYAK ticker sekaligus dalam SATU request ke Yahoo Finance
    (yf.download), jauh lebih cepat & lebih hemat request dibanding loop
    yf.Ticker().history() satu per satu. Dipakai oleh endpoint screener.

    Return: dict {ticker: DataFrame}. Ticker yang gagal/data kosong tidak masuk dict.
    """
    if not tickers:
        return {}
    try:
        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            threads=True,
            progress=False
        )
    except Exception:
        return {}

    hasil = {}
    if data is None or data.empty:
        return hasil

    if len(tickers) == 1:
        # yf.download tidak membuat kolom multi-index kalau cuma 1 ticker
        t = tickers[0]
        df_bersih = data.dropna(how='all')
        if not df_bersih.empty:
            hasil[t] = df_bersih
        return hasil

    for t in tickers:
        try:
            df_t = data[t].dropna(how='all')
            if not df_t.empty:
                hasil[t] = df_t
        except Exception:
            continue
    return hasil


def _ambil_info_dengan_retry(saham):
    """
    Retry pengambilan .info sampai RETRY_PERCOBAAN_MAKSIMAL kali kalau Yahoo Finance
    gagal sesaat / balikin data kosong (transient error), sebelum benar-benar dianggap gagal.

    CATATAN: sebelumnya fungsi ini mewajibkan field 'trailingEps' ada, yang bikin saham
    kecil/kurang ter-cover Yahoo Finance (misal VAST) selalu gagal walau data harga &
    teknikalnya sebenarnya tersedia. Sekarang cukup ada SALAH SATU sumber harga acuan
    (previousClose / currentPrice / regularMarketPrice) — bagian fundamental yang hilang
    akan di-fallback ke nilai default di hitung_analisis_saham, bukan bikin seluruh
    analisis gagal total.
    """
    for percobaan in range(RETRY_PERCOBAAN_MAKSIMAL):
        try:
            info = saham.info
            if info and (info.get('previousClose') or info.get('currentPrice') or info.get('regularMarketPrice')):
                return info
        except Exception:
            pass
        if percobaan < RETRY_PERCOBAAN_MAKSIMAL - 1:
            time.sleep(RETRY_JEDA_DETIK)
    return None


# =========================================================================
# INDIKATOR TEKNIKAL & UTILITAS ANALISIS
# =========================================================================

def hitung_indikator_lengkap(df, period=14):
    """
    Menghitung RSI, Stochastic, MACD, ATR, ADX, dan DI+/DI- secara native.

    RSI, ATR, dan ADX memakai Wilder's smoothing (ewm alpha=1/period), BUKAN rata-rata
    rolling biasa. Ini formula standar industri — nilai yang dihasilkan konsisten dengan
    chart di TradingView/aplikasi sekuritas, sehingga ambang seperti ADX > 20 punya makna
    yang sama dengan literatur. Rolling mean biasa membuat ADX "loncat-loncat" (nilai lama
    keluar dari window sekaligus) dan umumnya lebih lambat mendeteksi tren baru.
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    df['RSI14'] = 100 - (100 / (1 + rs))

    df['L14'] = df['Low'].rolling(window=period).min()
    df['H14'] = df['High'].rolling(window=period).max()
    df['Stoch_K'] = 100 * ((df['Close'] - df['L14']) / (df['H14'] - df['L14'] + 1e-10))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    df['UpMove'] = df['High'].diff()
    df['DownMove'] = -df['Low'].diff()

    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0.0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0.0)

    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    tr_smooth = df['TR'].ewm(alpha=1 / period, adjust=False).mean()
    df['ATR14'] = tr_smooth

    plus_dm_smooth = df['+DM'].ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = df['-DM'].ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
    minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))

    df['+DI14'] = plus_di
    df['-DI14'] = minus_di

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df['ADX14'] = dx.ewm(alpha=1 / period, adjust=False).mean()

    # Chaikin Money Flow 20-periode: proxy "arus bandar" dari data harga+volume.
    # Mengukur apakah volume lebih banyak terjadi saat harga ditutup dekat HIGH
    # (tekanan beli / akumulasi) atau dekat LOW (tekanan jual / distribusi).
    # Bukan pengganti broker summary asli, tapi menangkap fenomena yang sama
    # tanpa butuh sumber data berbayar.
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-10)
    df['CMF20'] = (mfm * df['Volume']).rolling(window=20).sum() / (df['Volume'].rolling(window=20).sum() + 1e-10)

    return df


# =========================================================================
# PRE-FILTER: SCREENING RINGAN (CPU-only, tanpa HTTP ke Yahoo Finance)
# Dipakai oleh screener di routers.py untuk mengeliminasi saham yang jelas
# tidak lolos SEBELUM membuang waktu download .info per ticker. Indikator
# dihitung ulang dari DataFrame yang sudah di-batch — overhead CPU ~milidetik
# per saham, vs .info yang butuh ~1-3 detik HTTP per saham.
# =========================================================================

def pre_filter_oversold_swing(df):
    """
    Pre-filter teknikal untuk screener swing: cek kondisi oversold (f1_kondisi)
    hanya dari data harga (DataFrame), TANPA memanggil Yahoo Finance .info.
    Dipakai untuk mengeliminasi saham yang jelas tidak lolos sebelum membuang
    waktu download data fundamental per ticker.

    Return True jika saham POTENSIAL lolos filter oversold (layak analisis lanjut).
    """
    if df is None or df.empty or len(df) < 200:
        return False
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df = hitung_indikator_lengkap(df)
    t = df.iloc[-1]
    return bool(
        t['Close'] < t['EMA20'] and
        t['Close'] > t['EMA200'] and
        t['Stoch_D'] <= 20 and
        t['MACD'] < 0
    )


def pre_filter_momentum(df, ema_span_pendek=5, ema_span_panjang=10,
                         volume_multiplier=2.5, volume_lookback=35,
                         min_data_points=40):
    """
    Pre-filter teknikal untuk screener momentum (gorengan & fast-intraday):
    cek 3 kondisi filter (volume spike, bullish EMA, ADX explosive) hanya dari
    data harga, TANPA memanggil Yahoo Finance .info.

    Parameter bisa di-override untuk menyesuaikan timeframe/strategi yang berbeda
    (gorengan: EMA5/10, volume 2.5x; fast-intraday: EMA5/13, volume 2.0x).
    """
    if df is None or df.empty or len(df) < min_data_points:
        return False
    df = df.copy()
    df['_ema_pendek'] = df['Close'].ewm(span=ema_span_pendek, adjust=False).mean()
    df['_ema_panjang'] = df['Close'].ewm(span=ema_span_panjang, adjust=False).mean()
    df = hitung_indikator_lengkap(df)
    t = df.iloc[-1]

    vol_terakhir = t['Volume']
    vol_rata2 = df['Volume'].iloc[:-1].tail(volume_lookback).mean()

    is_volume_spike = vol_rata2 > 0 and vol_terakhir > (vol_rata2 * volume_multiplier)
    is_bullish = t['Close'] > t['_ema_pendek'] and t['_ema_pendek'] > t['_ema_panjang']
    is_explosive = t['ADX14'] > ADX_THRESHOLD_GORENGAN and t['+DI14'] > t['-DI14']

    return bool(is_volume_spike and is_bullish and is_explosive)


def bulatkan_ke_tick_idx(harga, ke_bawah=True):
    """
    Bulatkan harga ke fraksi harga (tick size) resmi BEI supaya harga rekomendasi
    bisa langsung dipakai antre order tanpa ditolak sistem broker.
    ke_bawah=True untuk harga beli (konservatif), False untuk harga jual.
    """
    if harga < 200:
        tick = 1
    elif harga < 500:
        tick = 2
    elif harga < 2000:
        tick = 5
    elif harga < 5000:
        tick = 10
    else:
        tick = 25
    if ke_bawah:
        return int(harga // tick * tick)
    return int(-(-harga // tick) * tick)


def nilai_kualitas_tren_adx(adx, plus_di, minus_di):
    """Klasifikasi kualitas tren berdasarkan kekuatan ADX + arah dominan DI."""
    # bool() eksplisit: nilai dari pandas adalah numpy.bool_ yang gagal diserialisasi JSON
    arah_bullish = bool(plus_di > minus_di)
    if adx >= ADX_TREN_EKSTREM:
        label = "SANGAT KUAT / EKSTREM ⚡ (waspada tren sudah matang, rawan pembalikan)"
    elif adx >= ADX_TREN_KUAT:
        label = "KUAT 🚀 (zona ideal untuk momentum trading)"
    elif adx >= ADX_TREN_MODERAT:
        label = "MODERAT 📈 (tren baru terbentuk, butuh konfirmasi volume)"
    else:
        label = "LEMAH / SIDEWAYS 💤 (ADX di bawah 20, tidak ada tren jelas)"

    tren_bagus = bool(adx >= ADX_TREN_MODERAT) and arah_bullish
    return {
        "tren_bagus_untuk_daytrading": tren_bagus,
        "kekuatan": label,
        "arah": "BULLISH (DI+ dominan)" if arah_bullish else "BEARISH (DI- dominan)"
    }


def interpretasi_arus_bandar_cmf(cmf):
    """
    Terjemahkan nilai Chaikin Money Flow jadi status arus bandar yang mudah dibaca.
    Ambang +/-0.05 dan +/-0.15 adalah konvensi umum interpretasi CMF.
    """
    if cmf is None or (isinstance(cmf, float) and pd.isna(cmf)):
        return {"cmf_20": None, "status": "TIDAK TERSEDIA", "penjelasan": "Data belum cukup untuk menghitung CMF 20-periode."}
    cmf = float(cmf)
    if cmf >= 0.15:
        status = "AKUMULASI KUAT 🟢🟢"
        penjelasan = "Volume terkonsentrasi saat harga ditutup dekat high — indikasi kuat ada pihak besar mengakumulasi."
    elif cmf >= 0.05:
        status = "AKUMULASI 🟢"
        penjelasan = "Tekanan beli lebih dominan dari tekanan jual dalam 20 periode terakhir."
    elif cmf > -0.05:
        status = "NETRAL ⚪"
        penjelasan = "Tidak ada dominasi arus dana yang jelas — pasar seimbang."
    elif cmf > -0.15:
        status = "DISTRIBUSI 🔴"
        penjelasan = "Tekanan jual lebih dominan — hati-hati, ada indikasi pihak besar mengurangi posisi."
    else:
        status = "DISTRIBUSI KUAT 🔴🔴"
        penjelasan = "Volume terkonsentrasi saat harga ditutup dekat low — indikasi kuat distribusi besar sedang berlangsung."
    return {"cmf_20": round(cmf, 3), "status": status, "penjelasan": penjelasan}


def hitung_rekomendasi_entry_daytrading(harga_sekarang, ema_pullback, atr, target_jual,
                                        adx, plus_di, minus_di, rsi=None, filter_lolos=True,
                                        alasan_filter_gagal=None, fee_persen=None,
                                        profit_bersih_min_persen=None, rr_min_persen=None,
                                        rsi_maks=None, atr_multiplier_sl=None):
    """
    Rekomendasi harga masuk untuk daytrading. HANYA aktif jika seluruh guard lolos:

    1. filter_lolos — kalau strategi pemanggil punya filter utama (mis. screening
       gorengan 3-syarat) dan filter itu GAGAL, tidak boleh ada rekomendasi harga
       (dulu bisa muncul entry padahal status filter GAGAL — menyesatkan).
    2. Kualitas tren ADX (>= 20 dan DI+ dominan).
    3. RSI belum jenuh beli ekstrem (< rsi_maks) — menolak
       entry di harga parabolik (kasus CBPE: RSI 90).
    4. Risk/reward dari entry terbaik minimal rr_min_persen —
       tren kuat dengan ruang profit sempit tetap setup buruk (kasus KBLV: RR ~1:1).

    Harga yang dihasilkan:
    - harga_entry_terbaik: antre limit beli di area pullback sehat.
    - harga_masuk_maksimal: batas atas harga beli yang masih layak — dibatasi TIGA hal:
      profit bersih minimal ke target, risk/reward minimal terhadap stop loss, dan
      TIDAK PERNAH di atas harga sekarang (mencegah 'mengejar' harga).

    Parameter fee_persen/profit_bersih_min_persen/rr_min_persen/rsi_maks/atr_multiplier_sl
    OPSIONAL — kalau None, pakai konstanta daytrading default (perilaku lama, tidak
    berubah). Dibuat overridable supaya strategi lain (fast-intraday) bisa reuse fungsi
    ini dengan ambang yang lebih ketat/sesuai timeframe-nya sendiri, tanpa duplikasi logic.
    """
    fee_persen = FEE_TRANSAKSI_TOTAL_PERSEN if fee_persen is None else fee_persen
    profit_bersih_min_persen = MIN_PROFIT_BERSIH_DAYTRADING_PERSEN if profit_bersih_min_persen is None else profit_bersih_min_persen
    rr_min_persen = MIN_RASIO_RISK_REWARD_DAYTRADING if rr_min_persen is None else rr_min_persen
    rsi_maks = RSI_MAKS_UNTUK_ENTRY_DAYTRADING if rsi_maks is None else rsi_maks
    atr_multiplier_sl = ATR_MULTIPLIER_SL if atr_multiplier_sl is None else atr_multiplier_sl

    kualitas = nilai_kualitas_tren_adx(adx, plus_di, minus_di)

    if not filter_lolos:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": f"❌ TIDAK ADA REKOMENDASI ENTRY — filter utama strategi GAGAL{f' ({alasan_filter_gagal})' if alasan_filter_gagal else ''}. Sinyal tren saja tidak cukup tanpa konfirmasi filter."
        }

    if not kualitas["tren_bagus_untuk_daytrading"]:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": "Tren ADX belum memenuhi syarat daytrading (butuh ADX >= 20 dengan DI+ dominan). Tidak ada rekomendasi harga masuk."
        }

    if rsi is not None and float(rsi) >= rsi_maks:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": (
                f"❌ SETUP DITOLAK — RSI {round(float(rsi), 1)} sudah jenuh beli ekstrem (ambang {int(rsi_maks)}). "
                f"Tren memang kuat, tapi masuk sekarang = membeli di pucuk euforia, rawan koreksi tajam. Tunggu pullback dan RSI mendingin."
            )
        }

    if not (atr and atr > 0) or harga_sekarang <= 0 or target_jual <= harga_sekarang:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "keterangan": "Tren ADX bagus, tapi target jual sudah di bawah/sama dengan harga sekarang (harga di puncak resisten) atau ATR tidak tersedia — ruang profit tidak cukup untuk entry baru."
        }

    entry_terbaik = min(harga_sekarang, max(ema_pullback, harga_sekarang - 0.5 * atr))
    entry_terbaik = bulatkan_ke_tick_idx(entry_terbaik, ke_bawah=True)
    stop_loss = bulatkan_ke_tick_idx(entry_terbaik - (atr_multiplier_sl * atr), ke_bawah=True)

    # --- GUARD RISK/REWARD ---
    risiko_poin = entry_terbaik - stop_loss
    reward_poin = target_jual - entry_terbaik
    rasio_rr = round(float(reward_poin) / risiko_poin, 2) if risiko_poin > 0 else 0.0
    estimasi_profit_persen = round(float((target_jual - entry_terbaik) / entry_terbaik) * 100 - fee_persen, 2)
    estimasi_rugi_persen = round(float((stop_loss - entry_terbaik) / entry_terbaik) * 100 - fee_persen, 2)

    if rasio_rr < rr_min_persen:
        return {
            "aktif": False,
            "kualitas_tren_adx": kualitas,
            "rasio_risk_reward": rasio_rr,
            "estimasi_profit_bersih_persen": estimasi_profit_persen,
            "estimasi_rugi_ke_sl_persen": estimasi_rugi_persen,
            "keterangan": (
                f"❌ SETUP BURUK, LEWATKAN — potensi profit ke target ({estimasi_profit_persen}%) tidak sepadan dengan potensi rugi ke stop loss ({estimasi_rugi_persen}%). "
                f"Rasio reward:risk hanya {rasio_rr}, minimal layak {rr_min_persen}. Tren kuat bukan berarti setup layak."
            )
        }

    # Batas masuk maksimal = yang PALING KETAT dari 3 syarat:
    # (1) profit bersih minimal ke target, (2) RR minimal terhadap SL, (3) harga sekarang
    faktor_biaya = 1 + (fee_persen + profit_bersih_min_persen) / 100
    maks_dari_profit = target_jual / faktor_biaya
    maks_dari_rr = (target_jual + rr_min_persen * stop_loss) / (1 + rr_min_persen)
    harga_masuk_maksimal = bulatkan_ke_tick_idx(min(maks_dari_profit, maks_dari_rr, harga_sekarang), ke_bawah=True)

    return {
        "aktif": True,
        "kualitas_tren_adx": kualitas,
        "harga_entry_terbaik": entry_terbaik,
        "harga_masuk_maksimal": harga_masuk_maksimal,
        "target_jual": bulatkan_ke_tick_idx(target_jual, ke_bawah=True),
        "stop_loss_disarankan": stop_loss,
        "rasio_risk_reward": rasio_rr,
        "estimasi_profit_bersih_dari_entry_terbaik_persen": estimasi_profit_persen,
        "estimasi_rugi_ke_sl_dari_entry_terbaik_persen": estimasi_rugi_persen,
        "keterangan": (
            f"Antre limit beli di area Rp{entry_terbaik} (entry terbaik). Batas masuk maksimal Rp{harga_masuk_maksimal} — "
            f"JANGAN mengejar di atas itu. Skenario dari entry terbaik: profit ke target Rp{bulatkan_ke_tick_idx(target_jual, ke_bawah=True)} = {estimasi_profit_persen}%, "
            f"rugi jika stop loss Rp{stop_loss} tersentuh = {estimasi_rugi_persen}% (rasio reward:risk {rasio_rr})."
        ),
        "asumsi": f"Fee transaksi bolak-balik {fee_persen}%, profit bersih minimal {profit_bersih_min_persen}%, rasio RR minimal {rr_min_persen}. Harga dibulatkan ke fraksi harga resmi BEI."
    }


def buat_penjelasan_teknikal(rsi, stoch_d, macd, macd_signal, adx, plus_di, minus_di,
                             harga_sekarang, ema20, ema50, ema200, is_volume_strong, cmf=None):
    """Penjelasan singkat per indikator dalam bahasa sederhana, dihasilkan dari nilai aktual."""
    if rsi >= 70:
        rsi_text = f"RSI {round(rsi, 1)} — jenuh beli (overbought), rawan koreksi jangka pendek."
    elif rsi <= 30:
        rsi_text = f"RSI {round(rsi, 1)} — jenuh jual (oversold), potensi mantul jika ada konfirmasi."
    elif rsi >= 50:
        rsi_text = f"RSI {round(rsi, 1)} — momentum cenderung bullish (di atas garis tengah 50)."
    else:
        rsi_text = f"RSI {round(rsi, 1)} — momentum cenderung bearish (di bawah garis tengah 50)."

    if stoch_d <= 20:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area oversold, harga tertekan dan berpotensi jenuh jual."
    elif stoch_d >= 80:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area overbought, hati-hati aksi ambil untung."
    else:
        stoch_text = f"Stochastic %D {round(stoch_d, 1)} — area netral, belum ada sinyal jenuh."

    if macd > macd_signal and macd > 0:
        macd_text = "MACD di atas garis sinyal dan di atas nol — momentum naik sedang berjalan."
    elif macd > macd_signal:
        macd_text = "MACD baru memotong ke atas garis sinyal tapi masih di bawah nol — indikasi awal pembalikan naik (early rebound)."
    elif macd < 0:
        macd_text = "MACD di bawah garis sinyal dan di bawah nol — tekanan turun masih dominan."
    else:
        macd_text = "MACD di atas nol tapi melemah ke bawah garis sinyal — momentum naik mulai mendingin."

    kualitas_adx = nilai_kualitas_tren_adx(adx, plus_di, minus_di)
    adx_text = f"ADX {round(adx, 1)} — kekuatan tren {kualitas_adx['kekuatan']}, arah {kualitas_adx['arah']}."

    if harga_sekarang > ema20 and ema20 > ema50 and harga_sekarang > ema200:
        ema_text = "Harga di atas EMA20 > EMA50 dan di atas EMA200 — struktur tren naik sehat di semua kerangka waktu."
    elif harga_sekarang > ema200:
        ema_text = "Harga masih di atas EMA200 (tren besar naik) tapi sedang koreksi terhadap EMA jangka pendek."
    else:
        ema_text = "Harga di bawah EMA200 — tren besar masih turun, sinyal beli apapun berisiko melawan arus."

    vol_text = ("Volume hari ini di atas 1.5x rata-rata 20 hari — pergerakan dikonfirmasi partisipasi pasar yang nyata."
                if is_volume_strong else
                "Volume di bawah ambang 1.5x rata-rata — pergerakan harga belum dikonfirmasi volume, rawan sinyal palsu.")

    hasil = {
        "rsi": rsi_text,
        "stochastic": stoch_text,
        "macd": macd_text,
        "adx": adx_text,
        "posisi_ema": ema_text,
        "volume": vol_text
    }
    if cmf is not None:
        arus = interpretasi_arus_bandar_cmf(cmf)
        hasil["arus_bandar"] = f"CMF {arus['cmf_20']} — {arus['status']}. {arus['penjelasan']}"
    return hasil


def cek_kekuatan_support_dan_resisten(df, harga_sekarang, window_hari=60, toleransi_persen=0.015):
    """Deteksi support memindai window_hari terakhir (default 60 hari), resisten pakai 120 hari."""
    df_window = df.tail(window_hari).copy()
    df_window['is_low'] = df_window['Low'] == df_window['Low'].rolling(window=10, center=True, min_periods=1).min()
    titik_terendah_historis = df_window[df_window['is_low']]['Low'].tolist()

    jumlah_sentuhan_support = 0
    area_support_kuat = 0

    for low_val in titik_terendah_historis:
        if low_val > 0 and abs(harga_sekarang - low_val) / low_val <= toleransi_persen:
            jumlah_sentuhan_support += 1
            area_support_kuat = int(low_val)

    if jumlah_sentuhan_support >= 3:
        klasifikasi_support = f"SANGAT KUAT 🔥 (Telah diuji {jumlah_sentuhan_support}x di area Rp{area_support_kuat})"
    elif jumlah_sentuhan_support == 2:
        klasifikasi_support = f"SEDANG 🛡️ (Telah diuji 2x di area Rp{area_support_kuat})"
    else:
        klasifikasi_support = "LEMAH / DINAMIS 💤 (Hanya mengandalkan garis EMA berjalan)"

    resisten_terdekat = int(df['High'].tail(120).max())
    return klasifikasi_support, resisten_terdekat, area_support_kuat


def cek_kondisi_market():
    """
    Filter makro: cek tren IHSG sebelum sinyal per-saham dipakai.

    Hasil di-cache selama CACHE_TTL_MARKET_DETIK (default 5 menit) untuk menghindari
    download ulang data ^JKSE yang sama saat banyak request masuk dalam waktu berdekatan
    (misalnya screener dengan 50+ saham yang masing-masing membutuhkan kondisi market).
    """
    now = time.time()
    if _cache_kondisi_market["data"] and (now - _cache_kondisi_market["waktu"]) < CACHE_TTL_MARKET_DETIK:
        return _cache_kondisi_market["data"]

    try:
        index = yf.Ticker(MARKET_INDEX_TICKER)
        df = index.history(period="6mo", auto_adjust=False)
        if df.empty or len(df) < 50:
            hasil = {"status": "TIDAK DIKETAHUI", "market_bullish": True, "keterangan": "Data indeks tidak tersedia, filter market dilewati"}
        else:
            df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
            harga_terakhir = float(df['Close'].iloc[-1])
            ema50_terakhir = float(df['EMA50'].iloc[-1])
            bullish = harga_terakhir > ema50_terakhir
            hasil = {
                "status": "BULLISH 📈" if bullish else "BEARISH 📉",
                "market_bullish": bool(bullish),
                "ihsg_saat_ini": round(harga_terakhir, 2),
                "ihsg_ema50": round(ema50_terakhir, 2)
            }
    except Exception:
        hasil = {"status": "TIDAK DIKETAHUI", "market_bullish": True, "keterangan": "Gagal mengambil data indeks, filter market dilewati"}

    _cache_kondisi_market["data"] = hasil
    _cache_kondisi_market["waktu"] = now
    return hasil


def cek_guardrail_fundamental(saham, info):
    """
    Circuit breaker fundamental sebagai pengganti stop-loss teknikal untuk average-down.

    PERBAIKAN: perbandingan dividen kini memperhitungkan bahwa bucket tahun berjalan
    mungkin BELUM LENGKAP (emiten belum selesai membagikan dividen tahunan). Sebelumnya
    perbandingan selalu memakai bucket terakhir vs sebelumnya, yang bisa menghasilkan
    false positive "dividen turun >30%" padahal hanya belum dibagikan sepenuhnya.
    """
    alasan = []
    aman = True
    try:
        divs = saham.dividends
        if not divs.empty:
            per_tahun = divs.resample('YE').sum()
            if len(per_tahun) >= 2:
                # Cek apakah bucket terakhir adalah tahun berjalan (belum lengkap).
                tahun_sekarang = datetime.datetime.now().year
                tahun_bucket_terakhir = per_tahun.index[-1].year

                dividen_baru = None
                dividen_lama = None

                if tahun_bucket_terakhir == tahun_sekarang:
                    # Tahun berjalan belum lengkap → bandingkan 2 tahun kalender
                    # SEBELUMNYA yang sudah close-book, bukan tahun berjalan.
                    if len(per_tahun) >= 3:
                        dividen_baru = float(per_tahun.iloc[-2])
                        dividen_lama = float(per_tahun.iloc[-3])
                else:
                    # Semua bucket sudah tahun lalu → aman bandingkan 2 terakhir
                    dividen_baru = float(per_tahun.iloc[-1])
                    dividen_lama = float(per_tahun.iloc[-2])

                if dividen_lama is not None and dividen_baru is not None and dividen_lama > 0:
                    if dividen_baru < dividen_lama * (1 - BATAS_TOLERANSI_PENURUNAN_DIVIDEN):
                        aman = False
                        turun_persen = round((1 - (dividen_baru / dividen_lama)) * 100, 1)
                        alasan.append(f"Dividen turun {turun_persen}% dari tahun sebelumnya")
    except Exception:
        pass

    eps = info.get('trailingEps', 0)
    if eps is not None and eps < 0:
        aman = False
        alasan.append("EPS negatif (perusahaan sedang mencatat rugi)")

    if not alasan:
        alasan.append("Tidak ada sinyal peringatan fundamental terdeteksi")

    return {"aman_untuk_average_down": aman, "alasan": alasan}


def hitung_zona_average_down(harga_sekarang, ema20, ema50, ema200, area_support_kuat):
    """
    Area akumulasi bertahap untuk strategi dividen tanpa cut loss.

    Hanya level yang berada DI BAWAH harga sekarang yang dipakai, diurutkan menurun
    (koreksi ringan -> dalam). Dulu level diambil mentah dari EMA20/50/200 — saat harga
    berada di bawah EMA200 (tren turun), "zona koreksi dalam" malah nyangkut DI ATAS
    harga sekarang (kasus BMRI & BBCA) dan menyuruh beli lebih mahal. Sekarang level
    yang tidak relevan dibuang, dan alokasi disesuaikan dengan jumlah level tersisa.
    """
    kandidat = []
    for nama, level in [("EMA20", ema20), ("EMA50", ema50), ("EMA200", ema200),
                        ("area support kuat", area_support_kuat)]:
        if level and level > 0 and level < harga_sekarang:
            level_int = int(level)
            if all(abs(level_int - k[1]) / harga_sekarang > 0.005 for k in kandidat):  # buang level dempet (<0.5%)
                kandidat.append((nama, level_int))

    kandidat.sort(key=lambda k: -k[1])
    kandidat = kandidat[:3]

    if not kandidat:
        return {
            "tersedia": False,
            "catatan": ("Tidak ada level acuan (EMA20/50/200/support) di bawah harga sekarang — harga sedang berada "
                        "di bawah semua garis acuan (tren turun dalam). Fokus TAHAN posisi yang ada; tunggu struktur "
                        "harga pulih sebelum merencanakan average down.")
        }

    alokasi_map = {1: [100], 2: [40, 60], 3: TRANCHE_ALLOKASI}
    alokasi = alokasi_map[len(kandidat)]
    label_koreksi = ["Koreksi ringan", "Koreksi sedang", "Koreksi dalam"]

    hasil = {"tersedia": True}
    for i, (nama, level) in enumerate(kandidat):
        hasil[f"tranche_{i + 1}"] = {
            "area_harga": level,
            "alokasi_persen": alokasi[i],
            "keterangan": f"{label_koreksi[i]}, dekat {nama}"
        }
    hasil["catatan"] = ("Alokasi bertahap ini asumsi guardrail_fundamental.aman_untuk_average_down bernilai true. "
                        "Jika false, evaluasi ulang sebelum menambah posisi.")
    return hasil


# =========================================================================
# STRATEGI 1: SWING-INVESTMENT DIVIDEN
# =========================================================================

def ambil_info_tanggal_dividen(info):
    """
    Coba ambil info tanggal terkait dividen dari Yahoo Finance.

    PENTING - keterbatasan yang perlu disadari:
    - Yahoo Finance TIDAK selalu punya data ini untuk saham IDX (cakupannya jauh lebih
      lengkap untuk saham AS). Field kosong BUKAN berarti saham tidak bagi dividen,
      bisa jadi cuma datanya tidak ter-cover Yahoo.
    - 'exDividendDate' dari Yahoo itu EX-DATE (tanggal saham mulai diperdagangkan TANPA
      hak dividen), BUKAN cum-date. Cum-date = hari bursa terakhir SEBELUM ex-date (hari
      terakhir kamu masih dapat hak dividen kalau beli saat itu). Di sini cum-date
      dihitung sebagai ESTIMASI (ex-date dikurangi 1 hari), bukan tanggal resmi.
    - Untuk kepastian jadwal cum-date, recording date (tanggal pencatatan), dan payment
      date (tanggal pembayaran) yang akurat, sumber otoritatifnya adalah pengumuman
      aksi korporasi resmi di idx.co.id atau KSEI — bukan Yahoo Finance.
    """
    ex_date_epoch = info.get('exDividendDate')
    if not ex_date_epoch:
        return {
            "tersedia": False,
            "catatan": "Data tanggal dividen tidak tersedia di Yahoo Finance untuk saham ini. Cek jadwal resmi di idx.co.id atau KSEI."
        }
    try:
        ex_date = datetime.datetime.utcfromtimestamp(ex_date_epoch).date()
        cum_date_perkiraan = ex_date - datetime.timedelta(days=1)
        return {
            "tersedia": True,
            "estimasi_cum_date": str(cum_date_perkiraan),
            "ex_dividend_date": str(ex_date),
            "catatan": "Cum-date di sini ESTIMASI (ex-date dikurangi 1 hari), bukan tanggal resmi. Verifikasi ke idx.co.id/KSEI sebelum mengambil keputusan berdasarkan tanggal ini."
        }
    except Exception:
        return {
            "tersedia": False,
            "catatan": "Gagal memproses data tanggal dividen dari Yahoo Finance."
        }


def hitung_analisis_saham(ticker_symbol: str, kondisi_market: dict = None, df_riwayat: pd.DataFrame = None):
    """
    df_riwayat: opsional, DataFrame historis yang SUDAH ditarik sebelumnya (misal lewat
    ambil_riwayat_batch di screener). Kalau None, fungsi ini akan fetch sendiri
    (dipakai untuk endpoint analisis 1 ticker berdiri sendiri).
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    info = _ambil_info_dengan_retry(saham)

    if not info:
        return None

    # --- A. PROSES DATA FUNDAMENTAL ---
    # `or 0/1.0`: Yahoo kadang mengisi field dengan None (bukan menghilangkan key),
    # sehingga .get(key, default) tetap mengembalikan None dan perbandingan angka crash
    eps = info.get('trailingEps') or 0
    pbv_ratio = info.get('priceToBook') or 0
    return_on_equity = info.get('returnOnEquity') or 0
    beta = info.get('beta') or 1.0

    total_dividen = info.get('dividendRate', 0)
    if total_dividen == 0 or total_dividen is None:
        divs = saham.dividends
        total_dividen = int(divs.resample('YE').sum().iloc[-1]) if not divs.empty else 0

    pe_acuan = PE_WAJAR_BANK if "Bank" in info.get('industry', '') else PE_WAJAR_UMUM
    # Fallback berjenjang untuk harga acuan (beberapa saham kecil tidak selalu punya
    # semua field ini terisi di Yahoo Finance)
    harga_acuan = info.get('previousClose') or info.get('currentPrice') or info.get('regularMarketPrice') or 0
    harga_wajar = int(eps * pe_acuan) if eps > 0 else int(harga_acuan)

    if total_dividen > 0:
        dividend_yield_persen = round((total_dividen / (harga_acuan or 1)) * 100, 2)
        if (dividend_yield_persen / 100) >= YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN:
            # Yield cukup signifikan untuk dijadikan basis valuasi utama
            harga_maks_layak_beli = int(total_dividen / TARGET_DIVIDEND_YIELD)
            status_dividen = f"LAYAK ({dividend_yield_persen}% Yield)"
            is_dividend_stock = True
        else:
            # Ada dividen tapi nominalnya receh - rumus dividend-yield di sini akan
            # menghasilkan angka tidak masuk akal (jauh di bawah harga wajar fundamental),
            # jadi fallback ke valuasi PE-based, dan diperlakukan sebagai bukan-dividend-stock
            # untuk keperluan guardrail risiko (nggak ada 'bantalan dividen' yang berarti).
            harga_maks_layak_beli = int(harga_wajar * 0.85)
            status_dividen = f"ADA DIVIDEN TAPI KECIL ({dividend_yield_persen}% Yield, di bawah ambang {int(YIELD_MINIMAL_UNTUK_VALUASI_DIVIDEN * 100)}%) ⚠️"
            is_dividend_stock = False
    else:
        harga_maks_layak_beli = int(harga_wajar * 0.85)
        status_dividen = "TIDAK ADA DIVIDEN ❌"
        is_dividend_stock = False

    # --- B. PROSES DATA TEKNIKAL ---
    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period="1y", auto_adjust=False)

    if df.empty or len(df) < 200:
        return None

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df = hitung_indikator_lengkap(df)

    terakhir = df.iloc[-1]
    harga_sekarang = int(terakhir['Close'])
    ema20, ema50, ema200 = int(terakhir['EMA20']), int(terakhir['EMA50']), int(terakhir['EMA200'])
    macd, stoch_d, rsi = terakhir['MACD'], terakhir['Stoch_D'], terakhir['RSI14']
    adx, plus_di, minus_di = terakhir['ADX14'], terakhir['+DI14'], terakhir['-DI14']
    cmf_terakhir = terakhir['CMF20'] if 'CMF20' in df.columns else None
    arus_bandar_cmf = interpretasi_arus_bandar_cmf(cmf_terakhir)

    volume_terakhir = int(terakhir['Volume'])
    volume_rata_rata = int(df['Volume'].tail(20).mean())
    is_volume_strong = volume_terakhir > (volume_rata_rata * 1.5)

    klasifikasi_support, resisten_terdekat, area_support_kuat = cek_kekuatan_support_dan_resisten(df, harga_sekarang)
    jarak_ke_resisten = round(((resisten_terdekat - harga_sekarang) / harga_sekarang) * 100, 2)

    # --- C. DETEKSI TEKANAN JUAL INSTITUSI & FORUM MATCH ---
    is_panic_selling = (harga_sekarang < ema50) and is_volume_strong
    status_arus_modal = "PANIC SELLING / INSTITUSI KELUAR ⚠️" if is_panic_selling else "ARUS KAS STABIL / NORMAL 👍"

    f1_kondisi = harga_sekarang < ema20 and harga_sekarang > ema200 and stoch_d <= 20 and macd < 0

    # --- DETEKSI MACD BULLISH CROSSOVER DI BAWAH GARIS NOL ("Early Rebound / Bottoming Signal") ---
    # Beda dengan f1_kondisi (yang cuma cek MACD < 0 secara statis), ini mendeteksi EVENT
    # garis MACD baru saja cross ke ATAS garis Signal-nya, sementara nilai MACD-nya sendiri
    # masih di bawah 0 - dianggap sinyal awal pembalikan sebelum tren beneran naik.
    histogram_sekarang = terakhir['MACD_Histogram']
    histogram_kemarin = df['MACD_Histogram'].iloc[-2] if len(df) >= 2 else histogram_sekarang
    macd_crossover_bullish = (histogram_kemarin <= 0) and (histogram_sekarang > 0)
    sinyal_macd_early_rebound = bool(macd_crossover_bullish and macd < 0)

    if f1_kondisi and sinyal_macd_early_rebound:
        status_forum_swing = "SANGAT AKTIF 🔥🔄 (Oversold + MACD Bullish Crossover)"
    elif f1_kondisi:
        status_forum_swing = "AKTIF 🔥"
    else:
        status_forum_swing = "TIDAK AKTIF 💤"

    f2_kondisi = (ema20 > ema50) and (rsi >= 50) and (adx > ADX_TREN_MODERAT) and (plus_di > minus_di) and (harga_sekarang > ema200) and is_volume_strong
    status_forum_day = "TREN SANGAT KUAT 🚀" if f2_kondisi else "TREN LEMAH / SIDEWAYS 💤"

    # --- REKOMENDASI HARGA ENTRY DAYTRADING (saat tren ADX bagus) ---
    # Target jual memakai resisten terdekat kalau masih ada ruang di atas harga,
    # kalau tidak (harga sudah di puncak) fallback ke proyeksi ATR.
    atr_terakhir = float(terakhir['ATR14']) if pd.notna(terakhir['ATR14']) else 0.0
    target_jual_daytrading = float(resisten_terdekat) if resisten_terdekat > harga_sekarang else harga_sekarang + (ATR_MULTIPLIER_TP * atr_terakhir)
    rekomendasi_daytrading = hitung_rekomendasi_entry_daytrading(
        harga_sekarang=harga_sekarang,
        ema_pullback=ema20,
        atr=atr_terakhir,
        target_jual=target_jual_daytrading,
        adx=adx, plus_di=plus_di, minus_di=minus_di,
        rsi=rsi
    )

    # --- D. LOGIKA GUARDRAIL & STRATEGI ---
    wajib_stop_loss = beta > 1.3 or not is_dividend_stock

    if wajib_stop_loss:
        alasan_risiko = "Beta tinggi" if beta > 1.3 else "tanpa bantalan dividen yang berarti"
        kategori_risiko = f"TINGGI ({alasan_risiko}; Beta: {round(beta, 2)}) 🔥"
        status_proteksi = "MURNI TRADING CEPAT (Wajib Stop Loss)"
        status_tren = "UPTREND SPEKULATIF 📈" if harga_sekarang > ema20 else "DOWNTREND SPEKULATIF 📉"
        rekomendasi = "WAIT/TRADING CEPAT - SET STOP LOSS DI GROWIN KETAT 3-5%!"
    else:
        kategori_risiko = f"RENDAH/AMAN (Beta: {round(beta, 2)}) 🛡️"
        status_proteksi = "AMAN UNTUK STRATEGI GABUNGAN (Bisa Tanpa Cut Loss)"

        if is_panic_selling:
            status_tren = "DOWNTREND DISKONTINU 📉"
            rekomendasi = "ANTRE BELI SUPER PASIF (Institusi sedang jualan, tunggu reda)"
        elif harga_sekarang > ema20 and ema20 > ema50:
            status_tren = "UPTREND 📈"
            rekomendasi = "BUY ON WEAKNESS (Antre Beli di GROWIN dekat EMA20)" if harga_sekarang <= (ema20 * 1.015) else "HOLD (Tunggu Koreksi Sehat)"
        elif harga_sekarang < ema20 and harga_sekarang > ema50:
            status_tren = "KOREKSI DALAM 📉"
            rekomendasi = "WAIT AND SEE (Tunggu Sentuh EMA50)"
        elif harga_sekarang < ema50:
            status_tren = "ZONA DISKON / BEARISH SEMANTARA 📉"
            rekomendasi = "ZONA SEROK / AKUMULASI (Harga Murah di Bawah EMA50)"
        else:
            status_tren = "KONSOLIDASI 📊"
            rekomendasi = "WAIT AND SEE"

    # Kualifikasi label tren: "UPTREND" versi EMA20/EMA50 itu kerangka jangka menengah.
    # Kalau harga masih di bawah EMA200, tren BESAR masih turun — tanpa kualifikasi ini
    # label terkesan lebih optimis dari kondisi sebenarnya (kasus BBCA).
    if "UPTREND" in status_tren and harga_sekarang < ema200:
        status_tren += " — JANGKA MENENGAH (harga masih di bawah EMA200, tren besar belum pulih ⚠️)"

    if f1_kondisi and not wajib_stop_loss and not is_panic_selling:
        rekomendasi = "BUY ON WEAKNESS ★★★ (Konfirmasi Oversold Forum Aktif!)"
    elif f2_kondisi and not wajib_stop_loss:
        rekomendasi = "STRONG BUY / MOMENTUM RIDE 🚀 (Konfirmasi Tren ADX Meledak!)"

    if kondisi_market is None:
        kondisi_market = cek_kondisi_market()
    if not kondisi_market.get("market_bullish", True) and "BUY" in rekomendasi and not wajib_stop_loss:
        rekomendasi = f"{rekomendasi} (⚠️ IHSG sedang BEARISH, pertimbangkan kurangi ukuran posisi)"

    # --- CROSS-CHECK VALUASI vs REKOMENDASI ---
    # Rekomendasi berbasis teknikal/momentum (f2_kondisi dkk) dan harga_wajar berbasis
    # fundamental itu 2 sistem yang independen - tanpa cross-check, sistem bisa dengan
    # pede bilang "STRONG BUY" sementara harga sebenarnya sudah jauh di atas harga wajar,
    # tanpa ada yang menandai kontradiksi ini. Field ini bikin kontradiksinya EKSPLISIT
    # alih-alih tersembunyi di 2 angka berbeda yang harus dibandingkan manual oleh user.
    premi_terhadap_wajar_persen = None
    peringatan_valuasi = None
    if harga_wajar > 0:
        premi_terhadap_wajar_persen = round(((harga_sekarang - harga_wajar) / harga_wajar) * 100, 2)
        if premi_terhadap_wajar_persen > 20 and ("BUY" in rekomendasi or "STRONG" in rekomendasi):
            peringatan_valuasi = (
                f"⚠️ Harga saat ini {premi_terhadap_wajar_persen}% DI ATAS estimasi harga wajar fundamental (Rp{harga_wajar}). "
                f"Sinyal beli di atas murni berbasis momentum teknikal, BUKAN valuasi murah — "
                f"risiko koreksi lebih besar kalau momentum berbalik arah."
            )
        elif "BUY" in rekomendasi and harga_maks_layak_beli > 0 and harga_sekarang > harga_maks_layak_beli:
            # Guard khusus strategi dividen: sinyal beli teknikal boleh muncul, tapi kalau
            # harga sudah di atas batas layak beli, yield efektif yang dikunci makin tipis —
            # dulu kasus ini lolos tanpa peringatan selama premi < 20% (kasus BBCA +12.6%).
            peringatan_valuasi = (
                f"⚠️ Harga saat ini (Rp{harga_sekarang}) DI ATAS batas maks layak beli (Rp{harga_maks_layak_beli}). "
                f"Sinyal beli ini murni soal TIMING teknikal — untuk strategi dividen, menambah posisi di harga ini "
                f"mengunci yield efektif yang lebih rendah. Pertimbangkan menunggu harga kembali ke bawah batas layak beli."
            )

    # --- E. TEXT PENJELASAN OTOMATIS ---
    posisi_pos = "di atas" if harga_sekarang > ema20 else "di bawah"
    vol_text = "disertai volume tinggi" if is_volume_strong else "dengan volume cenderung rendah"
    penjelasan_chart = f"Harga {ticker_symbol.upper()} (Rp{harga_sekarang}) berada {posisi_pos} garis acuan EMA 20 (Rp{ema20}). Pergerakan harian berjalan {vol_text}. Grafik menunjukkan kondisi {status_tren}. Arus institusi saat ini terdeteksi {status_arus_modal}."

    # --- F. ZONA AVERAGE DOWN + GUARDRAIL FUNDAMENTAL ---
    zona_average_down = None
    guardrail_fundamental = None
    if not wajib_stop_loss:
        zona_average_down = hitung_zona_average_down(harga_sekarang, ema20, ema50, ema200, area_support_kuat)
        guardrail_fundamental = cek_guardrail_fundamental(saham, info)
        if not guardrail_fundamental["aman_untuk_average_down"]:
            rekomendasi = f"{rekomendasi} | ⚠️ GUARDRAIL FUNDAMENTAL AKTIF: pertimbangkan HENTIKAN average down, cek alasan di field guardrail_fundamental"

    # --- PANDUAN STRATEGI (dibangun SETELAH zona average down supaya konsisten) ---
    # Dulu panduan selalu memakai template dividen (antre beli EMA20, serok EMA50) untuk
    # SEMUA saham — termasuk saham spekulatif non-dividen (kasus KBLV/CBPE) dan level
    # yang berada di atas harga sekarang. Sekarang templatenya mengikuti kategori saham.
    if wajib_stop_loss:
        if rekomendasi_daytrading.get("aktif"):
            panduan_saran_growin = (
                f"Saham ini kategori MURNI TRADING CEPAT (bukan saham dividen) — JANGAN average down. "
                f"Jika ingin trading: antre limit beli di area Rp{rekomendasi_daytrading['harga_entry_terbaik']}, "
                f"pasang stop loss otomatis Rp{rekomendasi_daytrading['stop_loss_disarankan']} BERSAMAAN dengan order beli, "
                f"target jual Rp{rekomendasi_daytrading['target_jual']}. Jangan beli di atas Rp{rekomendasi_daytrading['harga_masuk_maksimal']}."
            )
        else:
            panduan_saran_growin = (
                "Saham ini kategori MURNI TRADING CEPAT (bukan saham dividen) dan saat ini TIDAK ADA setup entry "
                "yang layak (lihat alasan di rekomendasi_daytrading). Jangan antre beli, jangan average down — tunggu setup berikutnya."
            )
    elif zona_average_down and zona_average_down.get("tersedia"):
        t1 = zona_average_down["tranche_1"]["area_harga"]
        panduan_saran_growin = (
            f"1. Pasang Auto Order Beli pertama di area koreksi terdekat Rp{t1} (lihat detail bertahap di zona_average_down). "
            f"2. Alokasikan dana bertahap sesuai tranche — jangan habiskan peluru di satu level. "
            f"3. Set jaring jual otomatis Take Profit GTC di atap resisten Rp{resisten_terdekat}."
        )
    else:
        panduan_saran_growin = (
            f"Harga sedang berada di bawah semua garis acuan (tren turun dalam) — belum ada zona beli bertahap yang sehat. "
            f"Fokus TAHAN posisi yang ada selama guardrail fundamental aman, dan set jaring jual Take Profit GTC di resisten Rp{resisten_terdekat} bila ingin mengurangi posisi saat pantulan."
        )

    # --- G. SARAN MANAJEMEN RISIKO POSISI (stateless, cuma saran batas, bukan tracking) ---
    # Saham kategori spekulatif (wajib stop loss) memakai batas alokasi gorengan yang
    # lebih kecil, bukan batas 15% saham dividen (dulu selalu 15% — kasus KBLV/CBPE).
    if wajib_stop_loss:
        manajemen_risiko = {
            "maks_alokasi_modal_persen": MAX_ALOKASI_GORENGAN_PERSEN,
            "keterangan": (f"Saham ini kategori spekulatif — batas alokasi mengikuti aturan trading cepat "
                           f"({MAX_ALOKASI_GORENGAN_PERSEN}% modal), BUKAN batas {MAX_ALOKASI_SWING_PERSEN}% saham dividen. "
                           "Sistem stateless: total alokasi lintas saham tetap kamu catat manual.")
        }
    else:
        manajemen_risiko = {
            "maks_alokasi_modal_persen": MAX_ALOKASI_SWING_PERSEN,
            "keterangan": "Saran batas alokasi modal ke SATU saham ini. Sistem tidak melacak posisi lain yang sudah kamu buka (stateless), jadi total across saham tetap perlu kamu catat manual."
        }

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "fundamental": {
            "harga_wajar": harga_wajar,
            "harga_maks_layak_beli": harga_maks_layak_beli,
            "pbv_ratio": round(pbv_ratio, 2) if pbv_ratio else "N/A",
            "return_on_equity": f"{round(return_on_equity * 100, 2)}%" if return_on_equity else "N/A",
            "status_dividen": status_dividen,
            "info_tanggal_dividen": ambil_info_tanggal_dividen(info)
        },
        "teknikal": {
            "status_tren": status_tren,
            "klasifikasi_lantai": klasifikasi_support,
            "target_atap_resisten": f"Rp{resisten_terdekat} (Potensi ruang kenaikan: +{jarak_ke_resisten}%)",
            "ema20": ema20, "ema50": ema50, "ema200": ema200,
            "rsi_14": round(rsi, 2), "stochastic_d": round(stoch_d, 2), "adx_strength": round(adx, 2),
            "plus_di_14": round(float(plus_di), 2), "minus_di_14": round(float(minus_di), 2),
            "macd": round(float(macd), 2),
            "macd_signal": round(float(terakhir['MACD_Signal']), 2),
            "macd_histogram": round(float(histogram_sekarang), 2),
            "status_arus_modal": status_arus_modal,
            "arus_bandar_cmf": arus_bandar_cmf,
            "konfirmasi_oversold_swing": status_forum_swing,
            "oversold_swing_aktif": bool(f1_kondisi),
            "macd_early_rebound_terdeteksi": sinyal_macd_early_rebound,
            "konfirmasi_daytrading_adx": status_forum_day,
            "rekomendasi_daytrading": rekomendasi_daytrading,
            "penjelasan_indikator": buat_penjelasan_teknikal(
                rsi, stoch_d, macd, float(terakhir['MACD_Signal']), adx, plus_di, minus_di,
                harga_sekarang, ema20, ema50, ema200, is_volume_strong, cmf=cmf_terakhir
            ),
            "penjelasan_chart": penjelasan_chart,
            "panduan_saran_growin": panduan_saran_growin
        },
        "guardrail_proteksi": {
            "kategori_risiko": kategori_risiko,
            "aturan_akun": status_proteksi,
            "wajib_stop_loss": wajib_stop_loss
        },
        "kondisi_market": kondisi_market,
        "zona_average_down": zona_average_down,
        "guardrail_fundamental": guardrail_fundamental,
        "manajemen_risiko": manajemen_risiko,
        "premi_terhadap_harga_wajar_persen": premi_terhadap_wajar_persen,
        "peringatan_valuasi": peringatan_valuasi,
        "rekomendasi_akhir": rekomendasi
    }


# =========================================================================
# STRATEGI 2: SCREENER MOMENTUM GORENGAN
# =========================================================================

def hitung_momentum_gorengan(ticker_symbol: str, df_riwayat: pd.DataFrame = None):
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)

    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period="60d", interval="1h", auto_adjust=False)

    if df.empty or len(df) < 40:
        return None

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df = hitung_indikator_lengkap(df)

    terakhir = df.iloc[-1]
    harga_sekarang = int(terakhir['Close'])
    ema5, ema10 = terakhir['EMA5'], terakhir['EMA10']
    rsi, adx, plus_di, minus_di = terakhir['RSI14'], terakhir['ADX14'], terakhir['+DI14'], terakhir['-DI14']
    atr = terakhir['ATR14']
    arus_bandar_cmf = interpretasi_arus_bandar_cmf(terakhir['CMF20'] if 'CMF20' in df.columns else None)

    volume_terakhir = terakhir['Volume']
    volume_rata_rata = df['Volume'].iloc[:-1].tail(35).mean()

    is_volume_spike = volume_rata_rata > 0 and volume_terakhir > (volume_rata_rata * 2.5)
    is_bullish_momentum = harga_sekarang > ema5 and ema5 > ema10
    is_trend_explosive = adx > ADX_THRESHOLD_GORENGAN and plus_di > minus_di

    if pd.notna(atr) and atr > 0:
        cl_level = bulatkan_ke_tick_idx(harga_sekarang - (ATR_MULTIPLIER_SL * atr), ke_bawah=True)
        tp_level = bulatkan_ke_tick_idx(harga_sekarang + (ATR_MULTIPLIER_TP * atr), ke_bawah=True)
        metode_tp_sl = "ATR-based"
    else:
        cl_level = bulatkan_ke_tick_idx(harga_sekarang * 0.965, ke_bawah=True)
        tp_level = bulatkan_ke_tick_idx(harga_sekarang * 1.07, ke_bawah=True)
        metode_tp_sl = "Fixed % (fallback)"

    trailing_stop_saran = bulatkan_ke_tick_idx(ema5, ke_bawah=True) if harga_sekarang > ema5 else cl_level

    # Status filter ditentukan DULU — rekomendasi entry harus tunduk pada filter ini.
    # Dulu rekomendasi entry bisa muncul dengan harga lengkap padahal status GAGAL
    # (kasus KBLV) — aplikasi seperti bilang "jangan trading" dan "ini harga entrinya"
    # dalam satu output.
    filter_lolos = bool(is_volume_spike and is_bullish_momentum and is_trend_explosive)
    if filter_lolos:
        status_filter = "LOLOS SCREENING 🔥 (Ledakan ADX + Bandar Masuk!)"
    else:
        syarat_gagal = []
        if not is_volume_spike:
            syarat_gagal.append("volume belum meledak > 2.5x rata-rata")
        if not is_bullish_momentum:
            syarat_gagal.append("struktur EMA intraday belum bullish")
        if not is_trend_explosive:
            syarat_gagal.append("ADX/arah tren belum memenuhi")
        status_filter = "GAGAL 💤"
        alasan_gagal = ", ".join(syarat_gagal)

    # Rekomendasi harga masuk hanya saat filter LOLOS + semua guard lolos (RSI belum
    # overbought ekstrem, risk/reward layak). EMA5 = acuan pullback intraday.
    atr_val = float(atr) if pd.notna(atr) and atr > 0 else harga_sekarang * 0.02
    rekomendasi_entry = hitung_rekomendasi_entry_daytrading(
        harga_sekarang=harga_sekarang,
        ema_pullback=float(ema5),
        atr=atr_val,
        target_jual=float(tp_level),
        adx=float(adx), plus_di=float(plus_di), minus_di=float(minus_di),
        rsi=float(rsi) if pd.notna(rsi) else None,
        filter_lolos=filter_lolos,
        alasan_filter_gagal=None if filter_lolos else alasan_gagal
    )

    try:
        info = saham.info
        beta = info.get('beta', 1.8) if info else 1.8
    except Exception:
        beta = 1.8

    manajemen_risiko = {
        "maks_alokasi_modal_persen": MAX_ALOKASI_GORENGAN_PERSEN,
        "maks_posisi_bersamaan_disarankan": MAX_JUMLAH_GORENGAN_BERSAMAAN,
        "keterangan": "Saran batas per posisi gorengan. Sistem tidak melacak berapa posisi gorengan yang sudah kamu buka bersamaan (stateless) — kamu perlu jaga disiplin ini manual."
    }

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "status_filter": status_filter,
        "indikator": {
            "lonjakan_volume": f"{round(volume_terakhir / volume_rata_rata, 1)}x Lipat" if volume_rata_rata > 0 else "N/A",
            "rsi_momentum": round(rsi, 2),
            "adx_power": round(adx, 2),
            "kualitas_tren_adx": nilai_kualitas_tren_adx(float(adx), float(plus_di), float(minus_di)),
            "arus_bandar_cmf": arus_bandar_cmf,
            "atr_volatilitas": round(float(atr), 2) if pd.notna(atr) else "N/A",
            "tingkat_volatilitas_beta": round(beta, 2)
        },
        "penjelasan_indikator": {
            "volume": (f"Volume jam terakhir {round(volume_terakhir / volume_rata_rata, 1)}x lipat rata-rata — indikasi ada pihak besar masuk."
                       if is_volume_spike else "Belum ada lonjakan volume berarti (butuh > 2.5x rata-rata)."),
            "momentum": ("Harga di atas EMA5 > EMA10 — momentum intraday bullish."
                         if is_bullish_momentum else "Struktur EMA intraday belum bullish (harga belum di atas EMA5>EMA10)."),
            "adx": (f"ADX {round(float(adx), 1)} dengan DI+ dominan — tren intraday sedang meledak."
                    if is_trend_explosive else f"ADX {round(float(adx), 1)} — tren intraday belum cukup kuat/arah belum bullish."),
            "arus_bandar": f"CMF {arus_bandar_cmf['cmf_20']} — {arus_bandar_cmf['status']}. {arus_bandar_cmf['penjelasan']}"
        },
        "rekomendasi_entry_daytrading": rekomendasi_entry,
        # Bracket order hanya ditampilkan saat ada sinyal masuk yang sah — dulu TP/SL
        # tetap dihitung & tampil saat filter GAGAL, seolah-olah ada rencana trading.
        "bracket_order_growin": {
            "target_take_profit": tp_level,
            "batas_cut_loss": cl_level,
            "trailing_stop_saran": trailing_stop_saran,
            "metode": metode_tp_sl
        } if (filter_lolos and rekomendasi_entry.get("aktif")) else None,
        "manajemen_risiko": manajemen_risiko,
        "peringatan_keamanan": (
            "RESIKO EKSTREM! Pergerakan harga murni ledakan tren momentum intraday."
            if filter_lolos else
            "Saham kategori risiko ekstrem, dan saat ini TIDAK ADA sinyal masuk yang sah."
        ),
        "rekomendasi_aksi": (
            "DAY TRADING CEPAT - WAJIB LANGSUNG SET AUTO ORDER STOP LOSS DI GROWIN!"
            if (filter_lolos and rekomendasi_entry.get("aktif")) else
            (f"JANGAN TRADING SAHAM INI SEKARANG — filter momentum GAGAL ({alasan_gagal}). Tunggu seluruh syarat menyala bersamaan."
             if not filter_lolos else
             "JANGAN TRADING SAHAM INI SEKARANG — filter lolos tapi setup ditolak guard keamanan (lihat alasan di rekomendasi_entry_daytrading).")
        )
    }


# =========================================================================
# STRATEGI 3: FAST INTRADAY ALERT (bukan scalping asli — lihat catatan config.py)
# =========================================================================

def hitung_sinyal_fast_intraday(ticker_symbol: str, df_riwayat: pd.DataFrame = None):
    """
    Sinyal intraday cepat (15 menit) untuk trader yang mau horizon lebih pendek dari
    gorengan (jam) tapi masih dieksekusi manual. Watchlist-nya sengaja saham LIKUID
    (blue chip), bukan saham gorengan — profit tipis butuh spread kecil.

    PENTING: ini bukan scalping sungguhan. Data Yahoo Finance untuk saham IDX delay
    ~15-20 menit, dan alur WA->n8n->Vercel->Yahoo makan waktu beberapa detik. Anggap
    ini alat cari SETUP, eksekusi tetap manual & butuh kecepatan reaksi dari kamu.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)

    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period=PERIODE_DATA_FAST_INTRADAY, interval=INTERVAL_FAST_INTRADAY, auto_adjust=False)

    # Minimal 40 bar (~10 jam bursa dengan interval 15 menit) supaya indikator
    # (RSI/ADX/CMF 14-20 periode) punya cukup data warmup untuk stabil.
    if df.empty or len(df) < 40:
        return None

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df = hitung_indikator_lengkap(df)

    terakhir = df.iloc[-1]
    harga_sekarang = int(terakhir['Close'])
    ema5, ema13 = terakhir['EMA5'], terakhir['EMA13']
    rsi, adx, plus_di, minus_di = terakhir['RSI14'], terakhir['ADX14'], terakhir['+DI14'], terakhir['-DI14']
    atr = terakhir['ATR14']
    arus_bandar_cmf = interpretasi_arus_bandar_cmf(terakhir['CMF20'] if 'CMF20' in df.columns else None)

    volume_terakhir = terakhir['Volume']
    volume_rata_rata = df['Volume'].iloc[:-1].tail(JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY).mean()

    is_volume_spike = volume_rata_rata > 0 and volume_terakhir > (volume_rata_rata * VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY)
    is_bullish_momentum = harga_sekarang > ema5 and ema5 > ema13
    is_trend_explosive = adx > ADX_THRESHOLD_GORENGAN and plus_di > minus_di

    # Target jual: resisten lokal dalam window data yang diambil (bukan 120 hari
    # seperti swing — window fast-intraday jauh lebih pendek), fallback ke ATR.
    resisten_lokal = float(df['High'].tail(60).max())
    atr_terakhir = float(atr) if pd.notna(atr) and atr > 0 else harga_sekarang * 0.01
    target_jual = resisten_lokal if resisten_lokal > harga_sekarang else harga_sekarang + (ATR_MULTIPLIER_TP_FAST_INTRADAY * atr_terakhir)

    filter_lolos = bool(is_volume_spike and is_bullish_momentum and is_trend_explosive)
    if filter_lolos:
        status_filter = "LOLOS SCREENING ⚡ (Momentum 15 Menit Terkonfirmasi)"
        alasan_gagal = None
    else:
        syarat_gagal = []
        if not is_volume_spike:
            syarat_gagal.append(f"volume belum meledak > {VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY}x rata-rata")
        if not is_bullish_momentum:
            syarat_gagal.append("struktur EMA5/EMA13 belum bullish")
        if not is_trend_explosive:
            syarat_gagal.append("ADX/arah tren belum memenuhi")
        status_filter = "GAGAL 💤"
        alasan_gagal = ", ".join(syarat_gagal)

    # Reuse fungsi entry daytrading, tapi dengan ambang yang DIPERKETAT khusus
    # fast-intraday: profit minimal, RR minimal, dan ATR multiplier SL semua beda
    # dari gorengan/swing. Ini yang memastikan "cepat" tidak berarti "ceroboh" —
    # kalau target realistis dari ATR 15 menit terlalu kecil buat nutup fee + profit
    # minimal, setup ini DITOLAK, bukan dipaksakan.
    rekomendasi_entry = hitung_rekomendasi_entry_daytrading(
        harga_sekarang=harga_sekarang,
        ema_pullback=float(ema5),
        atr=atr_terakhir,
        target_jual=float(target_jual),
        adx=float(adx), plus_di=float(plus_di), minus_di=float(minus_di),
        rsi=float(rsi) if pd.notna(rsi) else None,
        filter_lolos=filter_lolos,
        alasan_filter_gagal=alasan_gagal,
        profit_bersih_min_persen=MIN_PROFIT_BERSIH_FAST_INTRADAY_PERSEN,
        rr_min_persen=MIN_RASIO_RISK_REWARD_FAST_INTRADAY,
        atr_multiplier_sl=ATR_MULTIPLIER_SL_FAST_INTRADAY
    )

    sl_fallback = bulatkan_ke_tick_idx(harga_sekarang - (ATR_MULTIPLIER_SL_FAST_INTRADAY * atr_terakhir), ke_bawah=True)
    tp_fallback = bulatkan_ke_tick_idx(harga_sekarang + (ATR_MULTIPLIER_TP_FAST_INTRADAY * atr_terakhir), ke_bawah=True)

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "status_filter": status_filter,
        "peringatan_bukan_scalping_asli": (
            "⚠️ Ini SINYAL, bukan eksekusi otomatis. Data Yahoo Finance untuk saham IDX "
            "delay ~15-20 menit — verifikasi harga real-time di aplikasi broker sebelum "
            "eksekusi. Disarankan max hold ~2 jam (8 bar 15-menit) kalau target/SL belum "
            "tersentuh, evaluasi ulang manual."
        ),
        "indikator": {
            "rsi_momentum": round(float(rsi), 2) if pd.notna(rsi) else "N/A",
            "adx_power": round(float(adx), 2) if pd.notna(adx) else "N/A",
            "kualitas_tren_adx": nilai_kualitas_tren_adx(float(adx), float(plus_di), float(minus_di)) if pd.notna(adx) else None,
            "arus_bandar_cmf": arus_bandar_cmf,
            "lonjakan_volume": f"{round(volume_terakhir / volume_rata_rata, 1)}x Lipat" if volume_rata_rata > 0 else "N/A",
            "atr_volatilitas": round(atr_terakhir, 2),
            "ema5": int(ema5), "ema13": int(ema13)
        },
        "penjelasan_indikator": {
            "volume": (f"Volume 15-menit terakhir {round(volume_terakhir / volume_rata_rata, 1)}x lipat rata-rata — indikasi ada dorongan beli mendadak."
                       if is_volume_spike else f"Belum ada lonjakan volume berarti (butuh > {VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY}x rata-rata)."),
            "momentum": ("Harga di atas EMA5 > EMA13 — momentum super jangka-pendek bullish."
                         if is_bullish_momentum else "Struktur EMA5/EMA13 belum bullish."),
            "adx": (f"ADX {round(float(adx), 1)} dengan DI+ dominan — tren jangka pendek sedang kuat."
                    if is_trend_explosive else f"ADX {round(float(adx), 1)} — tren jangka pendek belum cukup kuat/arah belum bullish."),
            "arus_bandar": f"CMF {arus_bandar_cmf['cmf_20']} — {arus_bandar_cmf['status']}. {arus_bandar_cmf['penjelasan']}"
        },
        "rekomendasi_entry_daytrading": rekomendasi_entry,
        "bracket_order_growin": {
            "target_take_profit": bulatkan_ke_tick_idx(target_jual, ke_bawah=True),
            "batas_cut_loss": sl_fallback if pd.notna(atr) else None,
            "maks_hold_bar_disarankan": MAX_HOLD_BARS_FAST_INTRADAY,
            "metode": "ATR-based (15 menit)"
        } if (filter_lolos and rekomendasi_entry.get("aktif")) else None,
        "rekomendasi_aksi": (
            "FAST INTRADAY - EKSEKUSI CEPAT, VERIFIKASI HARGA REAL-TIME DI BROKER, WAJIB SET STOP LOSS!"
            if (filter_lolos and rekomendasi_entry.get("aktif")) else
            (f"JANGAN TRADING SAHAM INI SEKARANG — filter momentum 15-menit GAGAL ({alasan_gagal}). Tunggu seluruh syarat menyala bersamaan."
             if not filter_lolos else
             "JANGAN TRADING SAHAM INI SEKARANG — filter lolos tapi setup ditolak guard keamanan (lihat alasan di rekomendasi_entry_daytrading, biasanya profit potensial terlalu tipis setelah fee).")
        )
    }


# =========================================================================
# STRATEGI 4: BSJP (Beli Sore Jual Pagi)
# =========================================================================

def evaluasi_sinyal_bsjp(df: pd.DataFrame, gain_min_persen=None, close_posisi_min=None,
                         volume_multiplier=None, value_min_rupiah=None, rsi_maks=None,
                         adx_min=None) -> pd.DataFrame:
    """
    Evaluasi sinyal BSJP di SETIAP baris data harian. Satu definisi kondisi dipakai
    bersama oleh hitung_sinyal_bsjp (sinyal hari ini) dan backtest_bsjp (simulasi
    historis) — tidak ada duplikasi logika yang bisa melenceng di antara keduanya.

    Syarat BSJP (standar komunitas, versi ketat):
    1. Kenaikan hari ini >= BSJP_GAIN_MIN_PERSEN
    2. Close di posisi atas range harian (>= 70% dari High-Low) — close kuat,
       bukan doji/pantulan lemah
    3. Volume >= BSJP_VOLUME_MULTIPLIER x rata-rata volume 20 hari SEBELUMNYA
       (rata-rata TIDAK termasuk hari ini — menghindari look-ahead bias)
    4. Nilai transaksi (Close x Volume) >= BSJP_VALUE_MIN_RUPIAH (likuiditas)
    5. Harga > SMA5 > SMA10 (momentum naik jangka pendek utuh)
    6. RSI < BSJP_RSI_MAKS (tolak kenaikan parabolik/jenuh)
    7. ADX > adx_min dengan DI+ dominan (tren terkonfirmasi)

    Semua ambang OPSIONAL — kalau None, pakai konstanta default (ketat) di config.py.
    Untuk varian LONGAR (cocok saham large-cap yang jarang meledak seperti BBRI):
    gain_min_persen=2, close_posisi_min=0.6, volume_multiplier=1.5, rsi_maks=100
    (guard RSI dinonaktifkan), adx_min=0 (guard ADX dinonaktifkan).
    """
    gain_min_persen = BSJP_GAIN_MIN_PERSEN if gain_min_persen is None else gain_min_persen
    close_posisi_min = BSJP_CLOSE_POSISI_RANGE_MIN if close_posisi_min is None else close_posisi_min
    volume_multiplier = BSJP_VOLUME_MULTIPLIER if volume_multiplier is None else volume_multiplier
    value_min_rupiah = BSJP_VALUE_MIN_RUPIAH if value_min_rupiah is None else value_min_rupiah
    rsi_maks = BSJP_RSI_MAKS if rsi_maks is None else rsi_maks
    adx_min = BSJP_ADX_MIN if adx_min is None else adx_min

    df = df.copy()
    df['SMA5'] = df['Close'].rolling(5).mean()
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['prev_close'] = df['Close'].shift(1)
    df['VolMA20_sebelum'] = df['Volume'].shift(1).rolling(20).mean().replace(0, np.nan)
    df = hitung_indikator_lengkap(df)

    rentang = (df['High'] - df['Low']).replace(0, np.nan)
    df['posisi_close_range'] = (df['Close'] - df['Low']) / rentang
    df['return_persen'] = (df['Close'] / df['prev_close'] - 1) * 100
    df['rasio_volume'] = df['Volume'] / df['VolMA20_sebelum']
    df['nilai_transaksi'] = df['Close'] * df['Volume']

    sinyal = (
        (df['return_persen'] >= gain_min_persen) &
        (df['posisi_close_range'] >= close_posisi_min) &
        (df['rasio_volume'] >= volume_multiplier) &
        (df['nilai_transaksi'] >= value_min_rupiah) &
        (df['Close'] > df['SMA5']) & (df['SMA5'] > df['SMA10']) &
        (df['RSI14'] < rsi_maks) &
        (df['ADX14'] > adx_min) & (df['+DI14'] > df['-DI14'])
    )
    df['sinyal_bsjp'] = sinyal
    return df


def hitung_statistik_gap_bsjp(df: pd.DataFrame, target_persen: float = None) -> dict:
    """
    Statistik GAP historis: untuk semua hari sinyal BSJP di masa lalu, berapa
    return jika beli di close hari sinyal lalu jual di open hari berikutnya
    (persis aturan BSJP). Ini basis data empiris untuk ekspektasi besok pagi —
    bukan jaminan, tapi frekuensi & distribusi sinyal ini di saham tersebut.

    target_persen: patokan "gap cukup untuk langsung jual" (default = BSJP_TARGET_PERSEN).
    """
    target_persen = BSJP_TARGET_PERSEN if target_persen is None else target_persen

    gap_list = []
    for i in range(len(df) - 1):
        val = df['sinyal_bsjp'].iloc[i]
        if not (pd.notna(val) and val):
            continue
        close = float(df['Close'].iloc[i])
        open_next = float(df['Open'].iloc[i + 1])
        if close <= 0:
            continue
        gap_list.append(round((open_next / close - 1) * 100, 2))

    if not gap_list:
        return {
            "jumlah_sinyal_sebelumnya": 0,
            "keterangan": "Belum ada sinyal BSJP pada periode data ini — tunggu momentum, jangan paksakan entry."
        }

    arr = np.array(gap_list)
    return {
        "jumlah_sinyal_sebelumnya": len(gap_list),
        "probabilitas_gap_up_persen": round(float((arr > 0).mean()) * 100, 2),
        "probabilitas_open_diatas_target_persen": round(float((arr >= target_persen).mean()) * 100, 2),
        "rata_rata_gap_persen": round(float(arr.mean()), 2),
        "median_gap_persen": round(float(np.median(arr)), 2),
        "gap_terburuk_persen": round(float(arr.min()), 2),
        "gap_terbaik_persen": round(float(arr.max()), 2),
        "catatan": (
            "⚠️ Sampel < 20 sinyal — hasil belum statistically robust, perlakukan sebagai indikasi awal saja."
            if len(gap_list) < 20 else
            "Sampel cukup (> 20 sinyal) — distribusi gap historis bisa dijadikan acuan ekspektasi kasar."
        )
    }


def hitung_sinyal_bsjp(ticker_symbol: str, df_riwayat: pd.DataFrame = None,
                       gain_min_persen=None, close_posisi_min=None,
                       volume_multiplier=None, value_min_rupiah=None, rsi_maks=None,
                       adx_min=None, target_persen=None):
    """
    Sinyal BSJP (Beli Sore Jual Pagi) berbasis data harian.

    Parameter ambang OPSIONAL (None = pakai default ketat di config.py) — lihat
    evaluasi_sinyal_bsjp untuk varian longgar yang cocok untuk BBRI.

    KETERBATASAN YANG HARUS DISADARI (dibaca sebelum eksekusi):
    - Data Yahoo Finance untuk saham IDX delay ~15-20 menit. Jika dipanggil
      sebelum pasar tutup (15.50 WIB), candle hari ini BELUM FINAL — volume &
      harga bisa berubah. Ambil data paling cepat ~15.30-15.45 WIB dan verifikasi
      harga real-time di aplikasi broker sebelum eksekusi.
    - Exit BSJP = jual di pembukaan besok (09.00-09.30 WIB) — harus manual,
      tidak bisa diotomasi dari data delayed.
    - Risiko utama = gap down semalam (berita global/domestik di luar jam bursa).
      Patuhi mental stop loss pagi; jangan ubah posisi ini jadi swing hold.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period=BSJP_PERIODE_DATA, interval="1d", auto_adjust=False)

    if df.empty or len(df) < 60:
        return None

    gain_min_persen = BSJP_GAIN_MIN_PERSEN if gain_min_persen is None else gain_min_persen
    close_posisi_min = BSJP_CLOSE_POSISI_RANGE_MIN if close_posisi_min is None else close_posisi_min
    volume_multiplier = BSJP_VOLUME_MULTIPLIER if volume_multiplier is None else volume_multiplier
    value_min_rupiah = BSJP_VALUE_MIN_RUPIAH if value_min_rupiah is None else value_min_rupiah
    rsi_maks = BSJP_RSI_MAKS if rsi_maks is None else rsi_maks
    adx_min = BSJP_ADX_MIN if adx_min is None else adx_min
    target_persen = BSJP_TARGET_PERSEN if target_persen is None else target_persen

    df = df.reset_index()
    df = evaluasi_sinyal_bsjp(
        df, gain_min_persen=gain_min_persen, close_posisi_min=close_posisi_min,
        volume_multiplier=volume_multiplier, value_min_rupiah=value_min_rupiah,
        rsi_maks=rsi_maks, adx_min=adx_min
    )

    terakhir = df.iloc[-1]
    if pd.isna(terakhir['Close']):
        return None

    harga_sekarang = int(terakhir['Close'])
    nilai_sinyal = terakhir['sinyal_bsjp']
    sinyal_hari_ini = bool(pd.notna(nilai_sinyal) and nilai_sinyal)

    detail_sinyal = {
        "kenaikan_hari_ini_persen": round(float(terakhir['return_persen']), 2) if pd.notna(terakhir['return_persen']) else "N/A",
        "posisi_close_di_range": round(float(terakhir['posisi_close_range']), 2) if pd.notna(terakhir['posisi_close_range']) else "N/A",
        "rasio_volume_vs_rata20": round(float(terakhir['rasio_volume']), 1) if pd.notna(terakhir['rasio_volume']) else "N/A",
        "nilai_transaksi_rupiah": int(terakhir['nilai_transaksi']) if pd.notna(terakhir['nilai_transaksi']) else "N/A",
        "rsi_14": round(float(terakhir['RSI14']), 2) if pd.notna(terakhir['RSI14']) else "N/A",
        "adx_14": round(float(terakhir['ADX14']), 2) if pd.notna(terakhir['ADX14']) else "N/A",
        "arah_tren_adx": "BULLISH (DI+ > DI-)" if (pd.notna(terakhir['+DI14']) and pd.notna(terakhir['-DI14']) and terakhir['+DI14'] > terakhir['-DI14']) else "BEARISH (DI- >= DI+)",
        "sma5": int(terakhir['SMA5']),
        "sma10": int(terakhir['SMA10']),
        "ambang_yang_dipakai": {
            "kenaikan_min_persen": gain_min_persen,
            "posisi_close_min": close_posisi_min,
            "volume_multiplier_min": volume_multiplier,
            "nilai_transaksi_min_rupiah": value_min_rupiah,
            "rsi_maks": rsi_maks,
            "adx_min": adx_min,
            "target_persen": target_persen
        }
    }

    if sinyal_hari_ini:
        status_filter = "LOLOS SCREENING BSJP 🐂 (Beli Sore Hari Ini, Jual Pagi Besok)"
        alasan_gagal = None
    else:
        syarat_gagal = []
        rp = terakhir['return_persen']
        if pd.notna(rp) and rp < gain_min_persen:
            syarat_gagal.append(f"kenaikan hanya {round(float(rp), 2)}% (butuh >= {gain_min_persen}%)")
        pcr = terakhir['posisi_close_range']
        if pd.notna(pcr) and pcr < close_posisi_min:
            syarat_gagal.append(f"close di posisi {round(float(pcr), 2)} range (butuh >= {close_posisi_min})")
        rv = terakhir['rasio_volume']
        if pd.notna(rv) and rv < volume_multiplier:
            syarat_gagal.append(f"volume {round(float(rv), 1)}x rata-rata (butuh >= {volume_multiplier}x)")
        nv = terakhir['nilai_transaksi']
        if pd.notna(nv) and nv < value_min_rupiah:
            syarat_gagal.append(f"nilai transaksi di bawah Rp{value_min_rupiah / 1e9:g} miliar (likuiditas kurang)")
        if not (pd.notna(terakhir['Close']) and pd.notna(terakhir['SMA5']) and pd.notna(terakhir['SMA10'])
                and terakhir['Close'] > terakhir['SMA5'] and terakhir['SMA5'] > terakhir['SMA10']):
            syarat_gagal.append("harga belum di atas SMA5 > SMA10 (momentum naik belum utuh)")
        rsi = terakhir['RSI14']
        if pd.notna(rsi) and rsi >= rsi_maks:
            syarat_gagal.append(f"RSI {round(float(rsi), 1)} sudah parabolik (>= {rsi_maks})")
        adx = terakhir['ADX14']
        if not (pd.notna(adx) and pd.notna(terakhir['+DI14']) and pd.notna(terakhir['-DI14'])
                and adx > adx_min and terakhir['+DI14'] > terakhir['-DI14']):
            syarat_gagal.append(f"ADX/arah tren belum memenuhi (butuh ADX > {adx_min} dengan DI+ dominan)")
        if not syarat_gagal:
            syarat_gagal.append("data indikator hari ini tidak lengkap (baris data belum stabil)")
        status_filter = "GAGAL 💤"
        alasan_gagal = syarat_gagal

    statistik_gap = hitung_statistik_gap_bsjp(df, target_persen=target_persen)

    if sinyal_hari_ini:
        harga_beli = bulatkan_ke_tick_idx(harga_sekarang, ke_bawah=True)
        target_jual = bulatkan_ke_tick_idx(harga_sekarang * (1 + target_persen / 100), ke_bawah=True)
        cut_loss = bulatkan_ke_tick_idx(harga_sekarang * (1 - BSJP_SL_PAGI_PERSEN / 100), ke_bawah=True)
        prob_up = statistik_gap.get("probabilitas_gap_up_persen")
        rekomendasi_bsjp = {
            "aktif": True,
            "aksi": "BELI SORE HARI INI sebelum pasar tutup (15.00-15.50 WIB) — JUAL BESOK PAGI di pembukaan (09.00-09.30 WIB)",
            "harga_beli_acuan": harga_beli,
            "target_jual_estimasi": target_jual,
            "harga_cut_loss_pagi": cut_loss,
            "mental_stop_loss_persen": BSJP_SL_PAGI_PERSEN,
            "estimasi_profit_bersih_persen": round(target_persen - FEE_TRANSAKSI_TOTAL_PERSEN, 2),
            "keterangan": (
                f"Antre beli di sekitar Rp{harga_beli} sore ini. Besok pagi jual di open (market order). "
                f"Jika gap up >= +{target_persen}%, langsung jual — estimasi bersih setelah fee "
                f"{FEE_TRANSAKSI_TOTAL_PERSEN}% = {round(target_persen - FEE_TRANSAKSI_TOTAL_PERSEN, 2)}%. "
                f"Jika gap down, cut di area Rp{cut_loss} ({BSJP_SL_PAGI_PERSEN}%) — JANGAN ditahan jadi swing."
            ),
            "asumsi": (
                "Hasil nyata = gap aktual besok pagi, bukan target. " +
                (f"Probabilitas gap up historis: {prob_up}%." if prob_up is not None else "Belum ada data gap historis.")
            )
        }
    else:
        rekomendasi_bsjp = {
            "aktif": False,
            "aksi": "JANGAN BELI SORE INI",
            "keterangan": "Syarat BSJP belum terpenuhi hari ini. Strategi ini hanya layak saat momentum kuat terbentuk di hari yang sama (kenaikan signifikan + volume meledak + close kuat)."
        }

    return {
        "saham": ticker_symbol.upper(),
        "harga_saat_ini": harga_sekarang,
        "status_filter": status_filter,
        "alasan_gagal": alasan_gagal,
        "detail_sinyal_hari_ini": detail_sinyal,
        "rekomendasi_bsjp": rekomendasi_bsjp,
        "statistik_gap_historis": statistik_gap,
        "peringatan_keamanan": (
            "⚠️ Data Yahoo Finance delay ~15-20 menit untuk saham IDX. Jika dipanggil sebelum pasar tutup, "
            "candle hari ini BELUM FINAL (volume bisa berubah) — jalankan paling cepat ~15.30-15.45 WIB dan "
            "verifikasi harga real-time di broker sebelum eksekusi. Exit besok pagi (09.00-09.30) dilakukan "
            "MANUAL; risiko utama adalah gap down semalam."
        )
    }


# =========================================================================
# STRATEGI 5: RANGE PAGI-SORE (JUAL PAGI, BELI SORE)
# Untuk pemegang saham (mis. BBRI) yang memanfaatkan pola intraday: harga
# cenderung menyentuh titik tertinggi di sesi pagi dan melemah di sesi sore.
# =========================================================================

def tabel_harian_pagi_sore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bangun tabel harian (prev_close, morning high, afternoon low, close) dari data
    intraday 15 menit, lengkap dengan peak/trough/spread relatif ke prev close.
    Dipakai bersama oleh hitung_analisis_range_pagi_sore dan backtest_range_pagi_sore
    sehingga definisi hari/pagi/sore konsisten di kedua sisi.

    Hari yang tidak punya bar pagi atau bar sore (mis. masih parsial) otomatis
    dibuang lewat dropna — jadi data hari berjalan yang belum lengkap tidak
    mencemari statistik.
    """
    df = df.copy()
    df = df.reset_index()
    kolom_waktu = 'Datetime' if 'Datetime' in df.columns else 'Date'
    waktu = pd.to_datetime(df[kolom_waktu])
    df['tanggal'] = waktu.dt.date
    df['jam_float'] = waktu.dt.hour + waktu.dt.minute / 60.0

    pagi = df[df['jam_float'] < RANGE_PAGI_SORE_JAM_PAGI]
    sore = df[df['jam_float'] >= RANGE_PAGI_SORE_JAM_SORE]
    if pagi.empty or sore.empty:
        return pd.DataFrame()

    harian = pd.DataFrame({
        'prev_close': df.groupby('tanggal')['Close'].last().shift(1),
        'morn_high': pagi.groupby('tanggal')['High'].max(),
        'aft_low': sore.groupby('tanggal')['Low'].min(),
        'close': df.groupby('tanggal')['Close'].last(),
    }).dropna()
    harian['peak_persen'] = (harian['morn_high'] / harian['prev_close'] - 1) * 100
    harian['trough_persen'] = (harian['aft_low'] / harian['prev_close'] - 1) * 100
    harian['spread_persen'] = (harian['morn_high'] - harian['aft_low']) / harian['prev_close'] * 100
    return harian


def hitung_analisis_range_pagi_sore(ticker_symbol: str, df_riwayat: pd.DataFrame = None,
                                    window_hari: int = None, persentil_jual: float = None,
                                    persentil_beli: float = None):
    """
    Analisis pola intraday 'Range Pagi-Sore' untuk saham yang SUDAH ANDA PEGANG:

    Berdasarkan N hari terakhir (default 30), hitung di harga berapa pasang order
    JUAL di sesi pagi (titik tertinggi pagi) dan di harga berapa pasang order BELI
    di sesi sore (titik terendah sore), lengkap dengan estimasi peluang terisi.

    Level jual = persentil ke-persentil_jual dari distribusi 'peak pagi' (semakin
    kecil persentil, level makin dekat ke prev close -> makin sering terisi).
    Level beli = persentil ke-persentil_beli dari distribusi 'trough sore'.

    CATATAN PENTING:
    - Ini SARAN level, bukan eksekusi otomatis. Pasang order manual di broker.
    - Data Yahoo Finance delay ~15-20 menit utk saham IDX — verifikasi harga
      real-time sebelum memutuskan.
    - IDX settlement T+2: menjual saham yang sudah Anda pegang lalu membeli kembali
      di hari yang sama diperbolehkan, tapi dana beli memakai cash terpisah (bukan
      hasil jual yang belum settle).
    - Jika order beli sore tidak terisi, JANGAN paksakan beli — pilih antara beli
      saat penutupan atau tahan cash.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    window_hari = RANGE_PAGI_SORE_WINDOW_HARI if window_hari is None else window_hari
    persentil_jual = RANGE_PAGI_SORE_PERSENTIL_JUAL if persentil_jual is None else persentil_jual
    persentil_beli = RANGE_PAGI_SORE_PERSENTIL_BELI if persentil_beli is None else persentil_beli

    saham = yf.Ticker(ticker)
    if df_riwayat is not None:
        df = df_riwayat.copy()
    else:
        df = saham.history(period=RANGE_PAGI_SORE_PERIODE, interval=RANGE_PAGI_SORE_INTERVAL, auto_adjust=False)

    if df is None or df.empty or len(df) < 100:
        return None

    df = df.reset_index()
    harian = tabel_harian_pagi_sore(df)
    if harian.empty or len(harian) < 15:
        return None

    # Jika sesi hari ini masih berjalan (belum 15.50 WIB), buang hari ini dari
    # statistik supaya trough/peak parsial tidak mencemari pola historis.
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=7)))
    tanggal_terakhir = harian.index[-1]
    sesi_sedang_berjalan = (
        tanggal_terakhir == now.date() and
        (now.hour < 15 or (now.hour == 15 and now.minute < 50))
    )
    if sesi_sedang_berjalan and len(harian) > 1:
        harian = harian.iloc[:-1]

    window = min(window_hari, len(harian))
    if window < 15:
        window = len(harian)
    harian_w = harian.tail(window)
    acuan = float(harian['close'].iloc[-1])  # close hari selesai terakhir

    level_jual_pct = float(harian_w['peak_persen'].quantile(persentil_jual))
    level_beli_pct = float(harian_w['trough_persen'].quantile(persentil_beli))
    level_jual_pct = max(level_jual_pct, 0.05)  # jangan sampai jual di bawah acuan

    harga_jual = bulatkan_ke_tick_idx(acuan * (1 + level_jual_pct / 100), ke_bawah=True)
    harga_beli = bulatkan_ke_tick_idx(acuan * (1 + level_beli_pct / 100), ke_bawah=False)

    hit_jual = round(float((harian_w['peak_persen'] >= level_jual_pct).mean()) * 100, 1)
    hit_beli = round(float((harian_w['trough_persen'] <= level_beli_pct).mean()) * 100, 1)
    kedua_terisi = round(float(
        ((harian_w['peak_persen'] >= level_jual_pct) & (harian_w['trough_persen'] <= level_beli_pct)).mean()
    ) * 100, 1)

    spread_bruto = round((harga_jual - harga_beli) / harga_beli * 100, 2)
    spread_bersih = round(spread_bruto - FEE_TRANSAKSI_TOTAL_PERSEN, 2)

    # --- Premise: kapan day-high & day-low terjadi? ---
    kolom_waktu = 'Datetime' if 'Datetime' in df.columns else 'Date'
    waktu = pd.to_datetime(df[kolom_waktu])
    df_tmp = df.copy()
    df_tmp['jam_float'] = waktu.dt.hour + waktu.dt.minute / 60.0
    idx_high = df_tmp.loc[df_tmp.groupby(pd.to_datetime(df_tmp[kolom_waktu]).dt.date)['High'].idxmax()]
    idx_low = df_tmp.loc[df_tmp.groupby(pd.to_datetime(df_tmp[kolom_waktu]).dt.date)['Low'].idxmin()]
    pct_high_pagi = round(float((idx_high['jam_float'] < RANGE_PAGI_SORE_JAM_PAGI).mean()) * 100, 1)
    pct_low_sore = round(float((idx_low['jam_float'] >= RANGE_PAGI_SORE_JAM_SORE).mean()) * 100, 1)

    def ringkas(s):
        return {
            "rata2_persen": round(float(s.mean()), 2),
            "median_persen": round(float(s.median()), 2),
            "p10_persen": round(float(s.quantile(0.10)), 2),
            "p90_persen": round(float(s.quantile(0.90)), 2),
        }

    if harga_beli >= harga_jual:
        status = "TIDAK LAYAK"
        keterangan = ("Spread antara level jual dan level beli tidak positif — pola range pagi-sore tidak "
                      "bisa dimanfaatkan dengan parameter ini. Coba persentil jual lebih kecil / beli lebih besar.")
    elif spread_bersih <= 0:
        status = "LAYAK TAPI TIPIS ⚠️"
        keterangan = (f"Spread bruto {spread_bruto}% hanya sedikit di atas fee {FEE_TRANSAKSI_TOTAL_PERSEN}% — "
                      "margin bersih tipis, pastikan level terisi dan jangan lupa biaya.")
    else:
        status = "LAYAK ✅"
        keterangan = (
            f"Pasang SELL LIMIT Rp{harga_jual} SEBELUM market buka (09.00 WIB). Jika terisi, pasang "
            f"BUY LIMIT Rp{harga_beli} di sesi sore (setelah 13.30 WIB). Estimasi: jual terisi {hit_jual}% hari, "
            f"beli terisi {hit_beli}% hari, dua-duanya terisi {kedua_terisi}% hari. Spread bruto {spread_bruto}%, "
            f"bersih setelah fee {FEE_TRANSAKSI_TOTAL_PERSEN}% = {spread_bersih}%. Jika beli tidak terisi, jangan "
            "paksakan beli — pilih beli saat penutupan atau tahan cash."
        )

    return {
        "saham": ticker_symbol.upper(),
        "strategi": "Range Pagi-Sore (Jual Pagi, Beli Sore)",
        "harga_acuan_prev_close": int(round(acuan)),
        "status": status,
        "jendela_hari_dipakai": int(window),
        "rekomendasi_order": {
            "harga_set_jual_pagi": harga_jual,
            "level_jual_persen": round(level_jual_pct, 2),
            "estimasi_terisi_pagi_persen": hit_jual,
            "harga_set_beli_sore": harga_beli,
            "level_beli_persen": round(level_beli_pct, 2),
            "estimasi_terisi_sore_persen": hit_beli,
            "estimasi_roundtrip_terisi_persen": kedua_terisi,
            "spread_bruto_persen": spread_bruto,
            "spread_bersih_setelah_fee_persen": spread_bersih,
            "keterangan": keterangan
        },
        "statistik_pola": {
            "persen_day_high_di_pagi": pct_high_pagi,
            "persen_day_low_di_sore": pct_low_sore,
            "peak_pagi": ringkas(harian_w['peak_persen']),
            "trough_sore": ringkas(harian_w['trough_persen']),
            "spread_pagi_sore": ringkas(harian_w['spread_persen']),
        },
        "contoh_hari_terakhir": [
            {
                "tanggal": str(idx),
                "prev_close": int(round(float(r['prev_close']))),
                "morning_high": int(round(float(r['morn_high']))),
                "afternoon_low": int(round(float(r['aft_low']))),
                "peak_persen": round(float(r['peak_persen']), 2),
                "trough_persen": round(float(r['trough_persen']), 2),
            }
            for idx, r in harian_w.tail(5).iterrows()
        ],
        "asumsi": f"Fee transaksi bolak-balik {FEE_TRANSAKSI_TOTAL_PERSEN}%. Harga dibulatkan ke fraksi harga resmi BEI.",
        "peringatan": (
            "⚠️ Data Yahoo Finance delay ~15-20 menit untuk saham IDX — verifikasi harga real-time di broker. "
            "IDX settlement T+2: dana beli sore memakai cash terpisah dari hasil jual pagi (belum settle). "
            "Pola ini statistik historis, bukan jaminan."
        )
    }
