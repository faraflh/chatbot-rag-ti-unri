"""Logic khusus kalender akademik: hitung hari libur antar-semester.

Diadaptasi 1:1 dari fungsi `hitung_hari_libur` di notebook asli.
Dipakai sebagai *bypass* RAG untuk pertanyaan spesifik tentang durasi liburan.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

_BULAN_INDO = {
    "januari": "01",
    "februari": "02",
    "febuari": "02",
    "maret": "03",
    "april": "04",
    "mei": "05",
    "juni": "06",
    "juli": "07",
    "agustus": "08",
    "september": "09",
    "oktober": "10",
    "november": "11",
    "desember": "12",
}


def get_tanggal_indo(now: Optional[datetime] = None) -> str:
    """Format tanggal hari ini dalam Bahasa Indonesia (e.g. '5 Mei 2026')."""
    if now is None:
        now = datetime.now()
    bulan = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    return f"{now.day} {bulan[now.month - 1]} {now.year}"


def terjemahkan_bulan_indo(tanggal_str: str) -> Optional[datetime]:
    """Parse string tanggal Indonesia ke datetime (e.g. '15 Agustus 2025')."""
    s = tanggal_str
    for nama_bulan, angka_bulan in _BULAN_INDO.items():
        if nama_bulan in s.lower():
            s = re.sub(nama_bulan, angka_bulan, s, flags=re.IGNORECASE)
            break

    try:
        parts = s.strip().split()
        if len(parts) == 3:
            return datetime.strptime(f"{parts[2]}-{parts[1]}-{parts[0]}", "%Y-%m-%d")
    except (ValueError, IndexError):
        return None
    return None


def hitung_hari_libur(
    teks_kalender: str,
    semester_awal: str,
    semester_tujuan: str,
    tahun_akademik: str,
) -> str:
    """Hitung selisih hari antara akhir perkuliahan semester_awal dan
    awal perkuliahan semester_tujuan, difilter berdasarkan tahun akademik."""
    pola_awal = (
        rf"semester: {semester_awal} \|.*kegiatan: Pelaksanaan Perkuliahan dan "
        r"Praktikum \| tanggal_mulai: (.*?) \| tanggal_selesai: (.*?) \|"
    )
    pola_tujuan = (
        rf"semester: {semester_tujuan} \|.*kegiatan: Pelaksanaan Perkuliahan dan "
        r"Praktikum \| tanggal_mulai: (.*?) \| tanggal_selesai: (.*?) \|"
    )

    semua_awal = list(re.finditer(pola_awal, teks_kalender, re.IGNORECASE))
    semua_tujuan = list(re.finditer(pola_tujuan, teks_kalender, re.IGNORECASE))

    if not semua_awal or not semua_tujuan:
        return (
            "Maaf, data tanggal perkuliahan untuk semester tersebut tidak "
            "ditemukan dalam dokumen."
        )

    tahun_valid = ["2025", "2026"] if "2025" in tahun_akademik else ["2026", "2027"]

    selisih_terbaik: Optional[int] = None
    hasil_terbaik = ""

    for match_awal in semua_awal:
        str_selesai_awal = match_awal.group(2).strip()
        if not any(thn in str_selesai_awal for thn in tahun_valid):
            continue
        tgl_selesai_awal = terjemahkan_bulan_indo(str_selesai_awal)
        if not tgl_selesai_awal:
            continue

        for match_tujuan in semua_tujuan:
            str_mulai_tujuan = match_tujuan.group(1).strip()
            tgl_mulai_tujuan = terjemahkan_bulan_indo(str_mulai_tujuan)
            if not tgl_mulai_tujuan:
                continue

            selisih = (tgl_mulai_tujuan - tgl_selesai_awal).days
            if selisih > 0 and (selisih_terbaik is None or selisih < selisih_terbaik):
                selisih_terbaik = selisih
                hasil_terbaik = (
                    f"Berdasarkan kalender akademik TA {tahun_akademik}:\n"
                    f"- Berakhirnya perkuliahan {semester_awal}: {str_selesai_awal}\n"
                    f"- Awal perkuliahan {semester_tujuan}: {str_mulai_tujuan}\n\n"
                    f"Total estimasi hari libur: **{selisih_terbaik} hari**."
                )

    if selisih_terbaik is not None:
        return hasil_terbaik
    return (
        f"Maaf, tidak ditemukan jadwal yang saling berhubungan untuk TA {tahun_akademik}."
    )


def deteksi_pertanyaan_libur(teks_user: str) -> Optional[tuple[str, str]]:
    """Deteksi pertanyaan tentang hari libur antar semester.

    Returns:
        (semester_awal, semester_tujuan) atau None jika bukan pertanyaan libur.
    """
    t = teks_user.lower()
    if "libur" not in t:
        return None
    if not ("genap" in t or "ganjil" in t):
        return None

    if "genap" in t and "ganjil" not in t:
        return ("Genap", "Ganjil")
    if "ganjil" in t and "genap" not in t:
        return ("Ganjil", "Genap")

    # Keduanya disebut → urutan kemunculan menentukan arah
    if t.find("genap") < t.find("ganjil"):
        return ("Genap", "Ganjil")
    return ("Ganjil", "Genap")


def deteksi_tahun_akademik(teks_user: str) -> Optional[str]:
    """Deteksi tahun akademik dari pertanyaan (mis. '25/26' atau '2025')."""
    t = teks_user.lower()
    if "25/26" in t or "2025" in t:
        return "2025/2026"
    if "26/27" in t or "2026" in t:
        return "2026/2027"
    return None
