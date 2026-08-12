# app/riwayat.py
"""
Riwayat digest harian + recap mingguan.

Penyimpanan memakai Upstash Redis (serverless KV, cocok untuk Vercel) lewat REST
API — TANPA dependency tambahan (cukup urllib stdlib). Env var yang dibutuhkan:
    UPSTASH_REDIS_REST_URL   (contoh: https://xxx.upstash.io)
    UPSTASH_REDIS_REST_TOKEN (token REST)

PENTING — desain GRACEFUL: kalau env var belum dipasang, semua fungsi tetap
berjalan tanpa error:
- simpan_snapshot_digest()  -> return {"tersimpan": False, "alasan": "KV belum dikonfigurasi"}
- ambil_riwayat_digest()    -> return {"tersedia": False, "riwayat": [], "peringatan": ...}
Jadi fitur notifikasi pagi/sore/ringkasan TIDAK rusak walau KV belum ada — hanya
bagian "riwayat harian" yang belum aktif sampai user memasang Upstash.

Struktur key:
    digest:{tanggal}  -> JSON snapshot ringkas digest pagi
    digest:dates      -> JSON list tanggal yang tersimpan (desc, max 30)

Recap mingguan menghitung dari snapshot-snapshot tsb: saham paling sering muncul
di Most Active, dan per strategi intraday (hari sinyal aktif, skor rata-rata,
saham paling sering masuk top-3).
"""
import json
import os
import urllib.parse
import urllib.request
from collections import Counter, defaultdict

KV_URL = os.environ.get("UPSTASH_REDIS_REST_URL", "").strip()
KV_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "").strip()

MAX_TANGGAL_TAMPAH = 30  # jaga ukuran KV: hanya simpan ~30 hari terakhir


def kv_terkonfigurasi() -> bool:
    """True jika env var Upstash sudah dipasang (URL + token)."""
    return bool(KV_URL and KV_TOKEN)


def _kv_request(method: str, key: str, value: str = None):
    """
    Panggil REST API Upstash. Return dict JSON response; None jika gagal.

    PENTING: Upstash REST memakai NAMA KOMANDO sebagai path segment
    (/set/{key} untuk tulis, /get/{key} untuk baca) — bukan nama method HTTP.
    Method HTTP hanya menandakan GET (tanpa body) vs POST (dengan body).
    """
    try:
        key_enc = urllib.parse.quote(key, safe="")
        komando = "set" if method.upper() == "POST" else "get"
        url = f"{KV_URL.rstrip('/')}/{komando}/{key_enc}"
        headers = {"Authorization": f"Bearer {KV_TOKEN}"}
        data = None
        if value is not None:
            data = value.encode("utf-8")
            headers["Content-Type"] = "text/plain"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _kv_set(key: str, value: str) -> bool:
    resp = _kv_request("POST", key, value)
    return bool(resp and resp.get("result") == "OK")


# Sentinel untuk membedakan "key tidak ada / kosong" (None) dari "request gagal"
# (lihat _kv_get) — supaya index tanggal tidak ditimpa saat error transient.
_KV_ERROR = object()


def _kv_get(key: str):
    resp = _kv_request("GET", key)
    if not resp:
        return _KV_ERROR  # request gagal (network/HTTP)
    return resp.get("result")  # None kalau key belum ada — itu NORMAL


def simpan_snapshot_digest(snapshot: dict) -> dict:
    """
    Simpan satu snapshot digest (ringkas) ke KV. Idempotent per tanggal
    (tanggal yang sama di-overwrite, tidak dobel).

    Return: {"tersimpan": bool, "tanggal": ..., "alasan": ...}
    """
    if not kv_terkonfigurasi():
        return {
            "tersimpan": False,
            "alasan": "KV belum dikonfigurasi — pasang UPSTASH_REDIS_REST_URL dan UPSTASH_REDIS_REST_TOKEN untuk mengaktifkan riwayat harian.",
        }
    tanggal = snapshot.get("tanggal")
    if not tanggal:
        return {"tersimpan": False, "alasan": "snapshot tidak punya field tanggal"}

    set_ok = _kv_set(f"digest:{tanggal}", json.dumps(snapshot, ensure_ascii=False))
    if not set_ok:
        return {"tersimpan": False, "alasan": "gagal menulis snapshot ke KV (periksa UPSTASH_REDIS_REST_URL/TOKEN)"}
    try:
        # Pertahankan daftar tanggal terurut desc (max MAX_TANGGAL_TAMPAH).
        # Baca index HANYA kalau GET sukses — kalau gagal transient, jangan
        # timpa index yang ada (hindari kehilangan riwayat tanggal lama).
        daftar_raw = _kv_get("digest:dates")
        if daftar_raw is _KV_ERROR:
            return {"tersimpan": True, "tanggal": tanggal, "catatan": "index tanggal tidak diperbarui (GET gagal), snapshot tetap tersimpan"}
        daftar = json.loads(daftar_raw) if daftar_raw else []
        if tanggal not in daftar:
            daftar.append(tanggal)
        daftar.sort(reverse=True)  # tanggal terbaru di depan
        daftar = daftar[:MAX_TANGGAL_TAMPAH]
        _kv_set("digest:dates", json.dumps(daftar))
        return {"tersimpan": True, "tanggal": tanggal}
    except Exception as e:
        return {"tersimpan": True, "tanggal": tanggal, "catatan": f"index tanggal gagal diperbarui: {e}"}
    except Exception as e:
        return {"tersimpan": False, "alasan": f"gagal simpan ke KV: {e}"}


def ambil_riwayat_digest(hari: int = 7) -> dict:
    """
    Ambil snapshot digest N hari terakhir (terbaru dulu).

    Return: {"tersedia": bool, "riwayat": [snapshot, ...], "peringatan": str|None}
    """
    if not kv_terkonfigurasi():
        return {
            "tersedia": False,
            "riwayat": [],
            "peringatan": "KV belum dikonfigurasi — pasang UPSTASH_REDIS_REST_URL dan UPSTASH_REDIS_REST_TOKEN untuk mengaktifkan riwayat harian.",
        }
    daftar_raw = _kv_get("digest:dates")
    daftar = json.loads(daftar_raw) if daftar_raw else []
    riwayat = []
    for tanggal in daftar[:hari]:
        raw = _kv_get(f"digest:{tanggal}")
        if raw:
            try:
                riwayat.append(json.loads(raw))
            except Exception:
                continue
    return {"tersedia": True, "riwayat": riwayat, "peringatan": None}


def recap_dari_riwayat(riwayat: list) -> dict:
    """
    Hitung recap mingguan dari daftar snapshot digest.

    Output:
      - jumlah_hari: berapa hari snapshot tersedia
      - saham_teratas_most_active: top saham paling sering muncul di Most Active
        (frekuensi + rata-rata % change hari)
      - per_strategi_intraday: untuk tiap strategi intraday, jumlah hari sinyal
        aktif, rata-rata skor terbaik, dan saham paling sering masuk top-3
    """
    if not riwayat:
        return {
            "jumlah_hari": 0,
            "saham_teratas_most_active": [],
            "per_strategi_intraday": [],
        }

    frek_most_active = Counter()
    pct_most_active = defaultdict(list)

    # per strategi: {kode: {"nama":.., "hari_aktif": n, "skor_total": n, "skor_jumlah": n, "pick_frek": Counter}}
    stat_strategi = defaultdict(lambda: {
        "nama": "", "hari_aktif": 0, "hari_total": 0,
        "skor_total": 0.0, "skor_jumlah": 0, "pick_frek": Counter(),
    })

    for snap in riwayat:
        for m in snap.get("most_active", []):
            t = m.get("ticker")
            if t:
                frek_most_active[t] += 1
                if m.get("pct_change_hari") is not None:
                    pct_most_active[t].append(m["pct_change_hari"])

        for lb in snap.get("rekomendasi_intraday", []):
            kode = lb.get("kode")
            if not kode:
                continue
            s = stat_strategi[kode]
            s["nama"] = lb.get("nama", kode)
            s["hari_total"] += 1
            terbaik = lb.get("terbaik") or []
            if terbaik and any(f.get("sinyal_aktif") for f in terbaik):
                s["hari_aktif"] += 1
            for f in terbaik[:3]:
                if f.get("skor") is not None:
                    s["skor_total"] += f["skor"]
                    s["skor_jumlah"] += 1
                if f.get("ticker"):
                    s["pick_frek"][f["ticker"]] += 1

    saham_teratas = []
    for t, n in frek_most_active.most_common(10):
        pcts = pct_most_active.get(t, [])
        saham_teratas.append({
            "ticker": t,
            "jumlah_hari_muncul": n,
            "rata_pct_change_hari": round(sum(pcts) / len(pcts), 2) if pcts else None,
        })

    per_strategi = []
    for kode, s in stat_strategi.items():
        top_picks = [{"ticker": t, "jumlah_hari": n} for t, n in s["pick_frek"].most_common(3)]
        per_strategi.append({
            "kode": kode,
            "nama": s["nama"],
            "hari_total": s["hari_total"],
            "hari_sinyal_aktif": s["hari_aktif"],
            "skor_rata2": round(s["skor_total"] / s["skor_jumlah"], 1) if s["skor_jumlah"] else None,
            "top_picks": top_picks,
        })
    per_strategi.sort(key=lambda x: -x["skor_rata2"] if x["skor_rata2"] is not None else -1)

    return {
        "jumlah_hari": len(riwayat),
        "saham_teratas_most_active": saham_teratas,
        "per_strategi_intraday": per_strategi,
    }
