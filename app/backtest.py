# app/backtest.py
"""
Modul backtest sederhana untuk kedua strategi. Tujuannya bukan simulasi eksekusi
order yang presisi (belum memperhitungkan slippage, fee broker, likuiditas order),
tapi untuk mengecek secara kasar: seberapa sering sinyal filter di services.py
benar-benar diikuti kenaikan harga di data historis, sebelum dipercaya dipakai
dengan uang sungguhan.

Keterbatasan yang perlu disadari:
- Backtest gorengan cuma bisa pakai window 60 hari data per-jam (limit yfinance
  untuk interval 1h), jadi jumlah sampel transaksi biasanya kecil dan HASILNYA
  TIDAK STATISTICALLY ROBUST. Anggap sebagai sanity-check kasar, bukan bukti kuat.
- Backtest swing pakai data harian sehingga sampel historisnya lebih panjang dan
  lebih bisa diandalkan dibanding backtest gorengan.
- Backtest gabungan (backtest_watchlist_*) menguji SELURUH watchlist sekaligus dan
  menggabungkan hasilnya, jauh lebih valid secara statistik dibanding 1 ticker saja
  karena tidak bergantung pada kebetulan performa 1 saham.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import pandas as pd
import numpy as np
from app.services import hitung_indikator_lengkap
from app.config import (
    ADX_THRESHOLD_GORENGAN, ATR_MULTIPLIER_SL, ATR_MULTIPLIER_TP,
    INDEX_BLUECHIP_UTAMA, WATCHLIST_GORENGAN,
    WATCHLIST_FAST_INTRADAY, INTERVAL_FAST_INTRADAY, PERIODE_DATA_FAST_INTRADAY,
    ATR_MULTIPLIER_SL_FAST_INTRADAY, ATR_MULTIPLIER_TP_FAST_INTRADAY,
    VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY, JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY,
    MAX_HOLD_BARS_FAST_INTRADAY
)


def _ringkas_hasil(trades, label_return="return_persen"):
    if not trades:
        return {
            "total_transaksi": 0,
            "keterangan": "Tidak ada sinyal yang terpicu pada periode data ini"
        }

    returns = [t[label_return] for t in trades]
    win_trades = [r for r in returns if r > 0]

    equity_curve = np.cumprod([1 + (r / 100) for r in returns])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = round(float(drawdown.min()) * 100, 2) if len(drawdown) else 0.0

    return {
        "total_transaksi": len(trades),
        "win_rate_persen": round((len(win_trades) / len(trades)) * 100, 2),
        "rata_rata_return_per_transaksi_persen": round(float(np.mean(returns)), 2),
        "total_return_gabungan_persen": round((float(equity_curve[-1]) - 1) * 100, 2),
        "max_drawdown_persen": max_drawdown,
        "detail_transaksi_terakhir": trades[-10:]
    }


def backtest_swing_dividen(ticker_symbol: str, tahun: int = 2, max_hold_hari: int = 30):
    """
    Simulasi sinyal Forum 1 (oversold swing): beli saat Harga < EMA20, Harga > EMA200,
    Stoch_D <= 20, MACD < 0. Keluar saat harga kembali menyentuh EMA20, atau setelah
    max_hold_hari (default 30 hari bursa) kalau belum tersentuh juga.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    df = saham.history(period=f"{tahun}y", auto_adjust=False)
    if df.empty or len(df) < 220:
        return None

    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df = hitung_indikator_lengkap(df)
    df = df.reset_index()
    df = df.dropna(subset=['EMA200', 'Stoch_D', 'MACD'])

    trades = []
    posisi_aktif = False
    harga_beli = 0.0
    hari_masuk = 0
    tanggal_beli = None

    for i in range(len(df)):
        row = df.iloc[i]
        if not posisi_aktif:
            sinyal_beli = (
                row['Close'] < row['EMA20'] and
                row['Close'] > row['EMA200'] and
                row['Stoch_D'] <= 20 and
                row['MACD'] < 0
            )
            if sinyal_beli:
                posisi_aktif = True
                harga_beli = float(row['Close'])
                hari_masuk = i
                tanggal_beli = row['Date'] if 'Date' in df.columns else i
        else:
            hari_berjalan = i - hari_masuk
            sinyal_jual = row['Close'] >= row['EMA20']
            if sinyal_jual or hari_berjalan >= max_hold_hari:
                harga_jual = float(row['Close'])
                return_persen = round(((harga_jual - harga_beli) / harga_beli) * 100, 2)
                trades.append({
                    "tanggal_beli": str(tanggal_beli.date()) if hasattr(tanggal_beli, 'date') else str(tanggal_beli),
                    "harga_beli": round(harga_beli, 2),
                    "tanggal_jual": str(row['Date'].date()) if 'Date' in df.columns else str(i),
                    "harga_jual": round(harga_jual, 2),
                    "hari_ditahan": int(hari_berjalan),
                    "alasan_keluar": "Sentuh EMA20" if sinyal_jual else "Batas maksimal hari tahan",
                    "return_persen": return_persen
                })
                posisi_aktif = False

    hasil = _ringkas_hasil(trades)
    hasil["saham"] = ticker_symbol.upper()
    hasil["strategi"] = "Swing Oversold (Forum 1)"
    hasil["periode_data_tahun"] = tahun
    hasil["catatan"] = "Backtest sederhana, belum memperhitungkan fee broker & slippage. Data harian, sampel relatif lebih dapat diandalkan dibanding backtest gorengan."
    return hasil


def backtest_gorengan_momentum(ticker_symbol: str, max_hold_jam: int = 12):
    """
    Simulasi sinyal momentum gorengan: masuk saat 3 kondisi filter terpenuhi
    (volume spike, bullish EMA5>EMA10, ADX explosive), keluar saat TP/SL (ATR-based)
    tersentuh atau setelah max_hold_jam.

    KETERBATASAN: data historis 1-jam dari yfinance dibatasi ~60 hari, jadi jumlah
    sinyal yang bisa dites terbatas dan hasilnya tidak statistically robust.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    df = saham.history(period="60d", interval="1h", auto_adjust=False)
    if df.empty or len(df) < 60:
        return None

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df = hitung_indikator_lengkap(df)
    df = df.reset_index()
    df = df.dropna(subset=['ADX14', 'ATR14']).reset_index(drop=True)

    trades = []
    posisi_aktif = False
    harga_beli = 0.0
    tp = 0.0
    sl = 0.0
    jam_masuk = 0
    waktu_beli = None

    for i in range(35, len(df)):
        row = df.iloc[i]
        if not posisi_aktif:
            volume_rata_rata = df['Volume'].iloc[max(0, i - 35):i].mean()
            is_volume_spike = volume_rata_rata > 0 and row['Volume'] > (volume_rata_rata * 2.5)
            is_bullish = row['Close'] > row['EMA5'] and row['EMA5'] > row['EMA10']
            is_explosive = row['ADX14'] > ADX_THRESHOLD_GORENGAN and row['+DI14'] > row['-DI14']

            if is_volume_spike and is_bullish and is_explosive:
                posisi_aktif = True
                harga_beli = float(row['Close'])
                atr = float(row['ATR14']) if pd.notna(row['ATR14']) and row['ATR14'] > 0 else harga_beli * 0.02
                tp = harga_beli + (ATR_MULTIPLIER_TP * atr)
                sl = harga_beli - (ATR_MULTIPLIER_SL * atr)
                jam_masuk = i
                waktu_beli = row['Datetime'] if 'Datetime' in df.columns else (row['index'] if 'index' in df.columns else i)
        else:
            jam_berjalan = i - jam_masuk
            harga_high = float(row['High'])
            harga_low = float(row['Low'])
            keluar = False
            alasan = ""
            harga_keluar = float(row['Close'])

            if harga_high >= tp:
                keluar = True
                alasan = "Take Profit"
                harga_keluar = tp
            elif harga_low <= sl:
                keluar = True
                alasan = "Cut Loss"
                harga_keluar = sl
            elif jam_berjalan >= max_hold_jam:
                keluar = True
                alasan = "Batas maksimal jam tahan"
                harga_keluar = float(row['Close'])

            if keluar:
                return_persen = round(((harga_keluar - harga_beli) / harga_beli) * 100, 2)
                trades.append({
                    "waktu_beli": str(waktu_beli),
                    "harga_beli": round(harga_beli, 2),
                    "harga_keluar": round(harga_keluar, 2),
                    "jam_ditahan": int(jam_berjalan),
                    "alasan_keluar": alasan,
                    "return_persen": return_persen
                })
                posisi_aktif = False

    hasil = _ringkas_hasil(trades)
    hasil["saham"] = ticker_symbol.upper()
    hasil["strategi"] = "Momentum Gorengan (ATR-based TP/SL)"
    hasil["periode_data"] = "60 hari terakhir, interval 1 jam (batas maksimal data intraday yfinance)"
    hasil["catatan"] = "⚠️ Sampel backtest kecil karena keterbatasan data intraday, hasil TIDAK statistically robust. Gunakan sebagai sanity-check kasar, bukan validasi final."
    return hasil


def backtest_fast_intraday(ticker_symbol: str, max_hold_bar: int = MAX_HOLD_BARS_FAST_INTRADAY):
    """
    Simulasi sinyal fast-intraday (15 menit): masuk saat 3 kondisi filter terpenuhi
    (volume spike, bullish EMA5>EMA13, ADX explosive), keluar saat TP/SL (ATR-based,
    kelipatan lebih ketat dari gorengan) tersentuh atau setelah max_hold_bar (default
    8 bar x 15 menit = ~2 jam).

    KETERBATASAN LEBIH PARAH dari backtest gorengan: data 15-menit yfinance dibatasi
    ~60 hari, dan strategi ini sendiri didesain untuk fetch window pendek (lihat
    PERIODE_DATA_FAST_INTRADAY). Jumlah sinyal yang bisa dites SANGAT terbatas —
    anggap ini sanity-check paling kasar dari semua backtest di aplikasi ini, bukan
    validasi. Untuk keyakinan lebih, jalankan backtest_watchlist_fast_intraday dan
    lihat apakah pola konsisten di banyak saham, bukan cuma 1.
    """
    if not ticker_symbol.endswith(".JK"):
        ticker = f"{ticker_symbol.upper()}.JK"
    else:
        ticker = ticker_symbol.upper()

    saham = yf.Ticker(ticker)
    # Untuk backtest (beda dari live signal) pakai window lebih panjang supaya ada
    # cukup sampel transaksi, dibatasi tetap oleh limit 60 hari yfinance untuk 15m.
    df = saham.history(period="60d", interval=INTERVAL_FAST_INTRADAY, auto_adjust=False)
    if df.empty or len(df) < 60:
        return None

    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    df = hitung_indikator_lengkap(df)
    df = df.reset_index()
    df = df.dropna(subset=['ADX14', 'ATR14']).reset_index(drop=True)

    trades = []
    posisi_aktif = False
    harga_beli = 0.0
    tp = 0.0
    sl = 0.0
    bar_masuk = 0
    waktu_beli = None

    for i in range(JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY, len(df)):
        row = df.iloc[i]
        if not posisi_aktif:
            volume_rata_rata = df['Volume'].iloc[max(0, i - JUMLAH_BAR_RATA_RATA_VOLUME_FAST_INTRADAY):i].mean()
            is_volume_spike = volume_rata_rata > 0 and row['Volume'] > (volume_rata_rata * VOLUME_SPIKE_MULTIPLIER_FAST_INTRADAY)
            is_bullish = row['Close'] > row['EMA5'] and row['EMA5'] > row['EMA13']
            is_explosive = row['ADX14'] > ADX_THRESHOLD_GORENGAN and row['+DI14'] > row['-DI14']

            if is_volume_spike and is_bullish and is_explosive:
                posisi_aktif = True
                harga_beli = float(row['Close'])
                atr = float(row['ATR14']) if pd.notna(row['ATR14']) and row['ATR14'] > 0 else harga_beli * 0.01
                tp = harga_beli + (ATR_MULTIPLIER_TP_FAST_INTRADAY * atr)
                sl = harga_beli - (ATR_MULTIPLIER_SL_FAST_INTRADAY * atr)
                bar_masuk = i
                waktu_beli = row['Datetime'] if 'Datetime' in df.columns else (row['index'] if 'index' in df.columns else i)
        else:
            bar_berjalan = i - bar_masuk
            harga_high = float(row['High'])
            harga_low = float(row['Low'])
            keluar = False
            alasan = ""
            harga_keluar = float(row['Close'])

            if harga_high >= tp:
                keluar = True
                alasan = "Take Profit"
                harga_keluar = tp
            elif harga_low <= sl:
                keluar = True
                alasan = "Cut Loss"
                harga_keluar = sl
            elif bar_berjalan >= max_hold_bar:
                keluar = True
                alasan = "Batas maksimal bar tahan"
                harga_keluar = float(row['Close'])

            if keluar:
                return_persen = round(((harga_keluar - harga_beli) / harga_beli) * 100, 2)
                trades.append({
                    "waktu_beli": str(waktu_beli),
                    "harga_beli": round(harga_beli, 2),
                    "harga_keluar": round(harga_keluar, 2),
                    "bar_15menit_ditahan": int(bar_berjalan),
                    "alasan_keluar": alasan,
                    "return_persen": return_persen
                })
                posisi_aktif = False

    hasil = _ringkas_hasil(trades)
    hasil["saham"] = ticker_symbol.upper()
    hasil["strategi"] = "Fast Intraday Alert (ATR-based TP/SL, 15 menit)"
    hasil["periode_data"] = "60 hari terakhir, interval 15 menit (batas maksimal data intraday yfinance)"
    hasil["catatan"] = "⚠️⚠️ Sampel backtest SANGAT kecil (keterbatasan data 15-menit + strategi cepat). Ini bukan scalping asli — anggap sanity-check paling kasar, bukan validasi. Verifikasi dengan backtest_watchlist_fast_intraday sebelum percaya pola apa pun."
    return hasil


# =========================================================================
# BACKTEST GABUNGAN SELURUH WATCHLIST (lebih valid secara statistik karena
# tidak bergantung pada kebetulan performa 1 saham saja)
# =========================================================================

def _gabungkan_hasil_backtest(hasil_per_saham: list):
    """Gabungkan beberapa hasil backtest per-saham jadi satu ringkasan tertimbang jumlah transaksi."""
    hasil_valid = [h for h in hasil_per_saham if h and h.get("total_transaksi", 0) > 0]

    if not hasil_valid:
        return {
            "jumlah_saham_diuji": 0,
            "total_transaksi_gabungan": 0,
            "keterangan": "Tidak ada transaksi tersimulasi di saham manapun pada periode ini"
        }

    total_transaksi = sum(h["total_transaksi"] for h in hasil_valid)
    win_rate_gabungan = round(
        sum(h["win_rate_persen"] * h["total_transaksi"] for h in hasil_valid) / total_transaksi, 2
    )
    rata2_return_gabungan = round(
        sum(h["rata_rata_return_per_transaksi_persen"] * h["total_transaksi"] for h in hasil_valid) / total_transaksi, 2
    )
    max_dd_terburuk = min(h["max_drawdown_persen"] for h in hasil_valid)

    return {
        "jumlah_saham_diuji": len(hasil_valid),
        "total_transaksi_gabungan": total_transaksi,
        "win_rate_gabungan_persen": win_rate_gabungan,
        "rata_rata_return_per_transaksi_persen": rata2_return_gabungan,
        "max_drawdown_terburuk_persen": max_dd_terburuk,
        "detail_per_saham": [
            {
                "saham": h["saham"],
                "total_transaksi": h["total_transaksi"],
                "win_rate_persen": h["win_rate_persen"],
                "total_return_gabungan_persen": h["total_return_gabungan_persen"]
            }
            for h in hasil_valid
        ]
    }


def backtest_watchlist_swing(tahun: int = 2, max_workers: int = 5):
    """Backtest strategi swing-oversold di SELURUH watchlist bluechip sekaligus, digabungkan."""
    hasil_per_saham = []

    def proses(ticker):
        symbol = ticker.replace(".JK", "")
        try:
            return backtest_swing_dividen(symbol, tahun=tahun)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in INDEX_BLUECHIP_UTAMA]
        for future in as_completed(futures):
            hasil_per_saham.append(future.result())

    ringkasan = _gabungkan_hasil_backtest(hasil_per_saham)
    ringkasan["strategi"] = "Swing Oversold (Forum 1) - Gabungan Watchlist"
    ringkasan["periode_data_tahun"] = tahun
    ringkasan["catatan"] = "Backtest tertimbang jumlah transaksi per saham. Lebih valid secara statistik dibanding backtest 1 ticker, tapi tetap belum memperhitungkan fee broker & slippage."
    return ringkasan


def backtest_watchlist_gorengan(max_workers: int = 5):
    """Backtest strategi momentum gorengan di SELURUH watchlist gorengan sekaligus, digabungkan."""
    hasil_per_saham = []

    def proses(ticker):
        symbol = ticker.replace(".JK", "")
        try:
            return backtest_gorengan_momentum(symbol)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in WATCHLIST_GORENGAN]
        for future in as_completed(futures):
            hasil_per_saham.append(future.result())

    ringkasan = _gabungkan_hasil_backtest(hasil_per_saham)
    ringkasan["strategi"] = "Momentum Gorengan - Gabungan Watchlist"
    ringkasan["periode_data"] = "60 hari terakhir per saham, interval 1 jam"
    ringkasan["catatan"] = "⚠️ Sampel tetap kecil karena keterbatasan data intraday 60 hari x jumlah saham gorengan yang sedikit. Sanity-check kasar, bukan validasi final."
    return ringkasan


def backtest_watchlist_fast_intraday(max_workers: int = 5):
    """Backtest strategi fast-intraday di SELURUH watchlist (saham likuid) sekaligus, digabungkan."""
    hasil_per_saham = []

    def proses(ticker):
        symbol = ticker.replace(".JK", "")
        try:
            return backtest_fast_intraday(symbol)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in WATCHLIST_FAST_INTRADAY]
        for future in as_completed(futures):
            hasil_per_saham.append(future.result())

    ringkasan = _gabungkan_hasil_backtest(hasil_per_saham)
    ringkasan["strategi"] = "Fast Intraday Alert - Gabungan Watchlist"
    ringkasan["periode_data"] = "60 hari terakhir per saham, interval 15 menit"
    ringkasan["catatan"] = "⚠️⚠️ Sampel SANGAT kecil (data intraday pendek x strategi cepat x watchlist terbatas). Backtest paling tidak robust di seluruh aplikasi ini — jangan jadikan basis keputusan tunggal."
    return ringkasan
