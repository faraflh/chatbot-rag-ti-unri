"""Logic khusus profil dosen: bypass RAG untuk pertanyaan 'list semua dosen'.

Top-k retriever (BM25 k=5 + Vector k=3) hanya mengembalikan 8 chunks. Kalau user
bertanya 'siapa saja dosen' / 'sebutkan semua dosen', LLM hanya melihat sebagian
data dan cenderung berhalusinasi (mengulang nama, mengarang nama). Modul ini
mendeteksi intent tersebut dan langsung mengembalikan daftar lengkap dari file
`dosen_rag_narasi.txt` (tanpa LLM), mirip dengan pendekatan `kalender_logic`
untuk pertanyaan libur.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


# Pola: "<NAMA + GELAR> adalah dosen dengan NIP ..."
_NAMA_RE = re.compile(r"^(.+?)\s+adalah\s+dosen", re.IGNORECASE)


# Kata kunci yang menandakan user MEMINTA daftar lengkap dosen.
_TRIGGERS = (
    "siapa saja",
    "semua dosen",
    "seluruh dosen",
    "daftar dosen",
    "daftar nama dosen",
    "daftar semua dosen",
    "list dosen",
    "berapa dosen",
    "berapa jumlah dosen",
    "total dosen",
    "sebutkan dosen",
    "sebutkan semua dosen",
    "sebutkan nama dosen",
    "sebutkan nama-nama dosen",
    "nama-nama dosen",
    "nama nama dosen",
)


# Kata kunci yang menandakan user MENCARI dosen spesifik (bukan list lengkap).
# Kalau salah satu muncul, biarkan flow RAG normal yang menjawab.
_VETO_QUALIFIERS = (
    "lektor",
    "asisten ahli",
    "guru besar",
    "kepakaran",
    "bidang ilmu",
    "bidang kepakaran",
    "email",
    "nip",
    "pengampu",
    "mata kuliah",
    "jabatan",
    "koordinator",
    "kaprodi",
    "koprodi",
    "ketua",
    "sekretaris",
)


def deteksi_pertanyaan_list_dosen(teks_user: str) -> bool:
    """Deteksi pertanyaan 'list semua dosen' (tanpa filter spesifik).

    Returns True kalau user meminta daftar lengkap dosen TI UNRI tanpa qualifier
    spesifik (misal: bukan 'siapa dosen yang Lektor Kepala').
    """
    t = teks_user.lower()
    if "dosen" not in t:
        return False
    if any(v in t for v in _VETO_QUALIFIERS):
        return False
    return any(trigger in t for trigger in _TRIGGERS)


def _extract_names_from_text(teks: str) -> List[str]:
    """Ekstrak nama-nama dosen dari teks narasi (urut, dedup).

    Pola: setiap baris yang match `<NAMA> adalah dosen ...` di-ambil <NAMA>-nya.
    """
    seen: set[str] = set()
    names: List[str] = []
    for line in teks.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _NAMA_RE.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def list_all_dosen(
    data_dir: Path, file_name: str = "dosen_rag_narasi.txt"
) -> Optional[List[str]]:
    """Baca file dosen langsung dari disk dan kembalikan list nama unik.

    Returns None kalau file tidak ditemukan.
    """
    path = data_dir / file_name
    if not path.exists():
        return None
    teks = path.read_text(encoding="utf-8")
    return _extract_names_from_text(teks)


def format_daftar_dosen(names: List[str]) -> str:
    """Format daftar nama dosen menjadi jawaban siap-tampil."""
    if not names:
        return (
            "Maaf, daftar dosen tidak ditemukan dalam dokumen yang tersedia."
        )
    body = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
    return (
        f"Berikut daftar **{len(names)} dosen** Program Studi Teknik Informatika "
        f"Universitas Riau:\n\n{body}"
    )
