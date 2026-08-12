# app/kajian_valuasi.py
"""
Kajian konsistensi VALUASI vs REKOMENDASI pada analisis swing (hitung_analisis_saham).

Mengukur dua masalah yang dikeluhkan user:
1. KONTRADIKSI: rekomendasi BUY/STRONG BUY muncul padahal harga > harga_maks_layak_beli.
2. HARGA WAJAR ANEH:
   - EPS <= 0: harga_wajar di-fallback ke harga pasar (premi selalu 0, terkesan "pas wajar").
   - harga_wajar melenceng jauh dari harga pasar / dari harga_maks_layak_beli.

Dijalankan:  python -m app.kajian_valuasi
Opsional ticker custom:  python -m app.kajian_valuasi BBRI TLKM BUMI
Seluruh saham BEI (941 ticker):  python -m app.kajian_valuasi --semua [--max-workers=10] [--batas=100]
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services import hitung_analisis_saham
from app.config import INDEX_BLUECHIP_UTAMA, WATCHLIST_GORENGAN, WATCHLIST_BSJP


def _nama_lengkap(t):
    return t if t.endswith(".JK") else f"{t.upper()}.JK"


def kajian_satu_saham(ticker_symbol: str) -> dict or None:
    """Ukur konsistensi valuasi vs rekomendasi untuk 1 saham."""
    data = hitung_analisis_saham(ticker_symbol)
    if not data:
        return None

    harga = data.get("harga_saat_ini") or 0
    wajar = data.get("fundamental", {}).get("harga_wajar")
    layak = data.get("fundamental", {}).get("harga_maks_layak_beli") or 0
    eps = data.get("fundamental", {}).get("trailing_eps")
    rekom = data.get("rekomendasi_akhir", "")
    premi_wajar = data.get("premi_terhadap_harga_wajar_persen")

    rekom_ada_buy = ("BUY" in rekom) or ("STRONG" in rekom)
    harga_di_atas_layak = bool(layak > 0 and harga > layak)

    # EPS <= 0 → harga_wajar tidak valid (fallback ke harga pasar / None)
    wajar_tidak_valid = bool(eps is not None and eps <= 0) or wajar is None

    # harga_wajar melenceng ekstrem dari harga pasar (> 100% beda)
    wajar_melenceng = None
    if isinstance(wajar, (int, float)) and wajar > 0 and harga > 0:
        beda = abs(harga - wajar) / wajar * 100
        wajar_melenceng = round(beda, 1)

    return {
        "saham": data.get("saham"),
        "harga": harga,
        "harga_wajar": wajar,
        "harga_maks_layak_beli": layak,
        "trailing_eps": eps,
        "rekomendasi": rekom,
        "premi_terhadap_wajar_persen": premi_wajar,
        "kontradiksi_buy_di_atas_layak": bool(rekom_ada_buy and harga_di_atas_layak),
        "wajar_tidak_valid": wajar_tidak_valid,
        "wajar_melenceng_persen": wajar_melenceng,
    }


def kajian_watchlist(tickers: list, max_workers: int = 5, progress_path: str = None) -> dict:
    """
    Kajian konsistensi valuasi untuk banyak saham sekaligus (paralel).

    progress_path: kalau diisi, hasil per ticker di-checkpoint ke file JSONL begitu
    selesai — kalau proses terputus (timeout/koneksi), hasil yang sudah jadi tidak
    hilang dan bisa dibaca ulang.
    """
    hasil = {}

    def proses(t):
        try:
            r = kajian_satu_saham(t)
        except Exception:
            r = None
        if progress_path:
            try:
                with open(progress_path, "a") as f:
                    f.write(json.dumps({t: r}, ensure_ascii=False, default=str) + "\n")
            except Exception:
                pass
        return t, r

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(proses, t) for t in tickers]
        for future in as_completed(futures):
            t, r = future.result()
            hasil[t] = r

    valid = {t: r for t, r in hasil.items() if r}
    n = len(valid)
    if n == 0:
        return {
            "jumlah_diproses": len(hasil),
            "jumlah_dianalisis": 0,
            "jumlah_gagal": len(hasil),
            "keterangan": "Tidak ada saham yang berhasil dianalisis.",
        }

    kontradiksi = [r for r in valid.values() if r["kontradiksi_buy_di_atas_layak"]]
    wajar_invalid = [r for r in valid.values() if r["wajar_tidak_valid"]]
    melenceng = [r for r in valid.values() if r["wajar_melenceng_persen"] is not None and r["wajar_melenceng_persen"] > 100]

    return {
        "jumlah_diproses": len(hasil),
        "jumlah_dianalisis": n,
        "jumlah_gagal": len(hasil) - n,
        "jumlah_kontradiksi_buy_di_atas_layak": len(kontradiksi),
        "persen_kontradiksi_persen": round(len(kontradiksi) / n * 100, 1),
        "daftar_kontradiksi": [
            {
                "saham": r["saham"],
                "harga": r["harga"],
                "harga_maks_layak_beli": r["harga_maks_layak_beli"],
                "rekomendasi": r["rekomendasi"],
            }
            for r in kontradiksi
        ],
        "jumlah_harga_wajar_tidak_valid": len(wajar_invalid),
        "daftar_wajar_tidak_valid": [
            {"saham": r["saham"], "eps": r["trailing_eps"], "harga_wajar": r["harga_wajar"], "harga": r["harga"]}
            for r in wajar_invalid
        ],
        "jumlah_harga_wajar_melenceng_>100pct": len(melenceng),
        "daftar_wajar_melenceng": [
            {"saham": r["saham"], "harga": r["harga"], "harga_wajar": r["harga_wajar"], "beda_persen": r["wajar_melenceng_persen"]}
            for r in melenceng
        ],
    }


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    max_workers = 5
    batas = None
    for a in sys.argv[1:]:
        if a.startswith("--max-workers="):
            try:
                max_workers = int(a.split("=", 1)[1])
            except ValueError:
                print(f"Argumen tidak valid (harus angka): {a}", file=sys.stderr)
        elif a.startswith("--batas="):
            try:
                batas = int(a.split("=", 1)[1])
            except ValueError:
                print(f"Argumen tidak valid (harus angka): {a}", file=sys.stderr)

    if "--semua" in sys.argv:
        from app.daftar_saham_bei import SEMUA_SAHAM_BEI
        ticker_list = list(SEMUA_SAHAM_BEI)
        laporan_file = "audit_bei_report.json"
        progress_file = "audit_bei_progress.jsonl"
    elif args:
        ticker_list = args
        laporan_file, progress_file = None, None
    else:
        gabungan = list(dict.fromkeys(INDEX_BLUECHIP_UTAMA + WATCHLIST_GORENGAN + WATCHLIST_BSJP))
        ticker_list = [_nama_lengkap(t) for t in gabungan]
        laporan_file, progress_file = None, None

    if batas:
        ticker_list = ticker_list[:batas]

    if progress_file and batas is None:
        # Run penuh: mulai dari progress bersih supaya file tidak menumpuk
        # hasil run sebelumnya / run sampel (--batas).
        open(progress_file, "w").close()

    print(f"Mengkaji {len(ticker_list)} saham (max_workers={max_workers})...", flush=True)
    laporan = kajian_watchlist(ticker_list, max_workers=max_workers, progress_path=progress_file)
    if laporan_file:
        with open(laporan_file, "w") as f:
            json.dump(laporan, f, ensure_ascii=False, indent=2, default=str)
        print(f"Laporan lengkap disimpan ke {laporan_file}", flush=True)
    print(json.dumps(laporan, ensure_ascii=False, indent=1, default=str), flush=True)
