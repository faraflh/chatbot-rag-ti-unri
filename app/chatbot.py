"""Orchestrator chatbot: gabungkan retrieval + kalender logic + LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

from .dosen_logic import (
    deteksi_pertanyaan_list_dosen,
    format_daftar_dosen,
    list_all_dosen,
)
from .kalender_logic import (
    deteksi_pertanyaan_libur,
    deteksi_tahun_akademik,
    get_tanggal_indo,
    hitung_hari_libur,
)
from .llm_backend import LLMBackend
from .prompts import SYSTEM_PROMPT, build_user_message_with_context
from .rag_pipeline import format_context, unique_sources


@dataclass
class ChatbotResponse:
    answer: str
    sources: List[str]
    used_special_logic: bool  # True jika dijawab oleh kalender_logic, bukan LLM


class LiburState:
    """Mini state machine untuk follow-up tahun akademik di pertanyaan libur."""

    def __init__(self) -> None:
        self.menunggu_jawaban: bool = False
        self.sem_awal: str = ""
        self.sem_tujuan: str = ""

    def reset(self) -> None:
        self.menunggu_jawaban = False
        self.sem_awal = ""
        self.sem_tujuan = ""


class Chatbot:
    """Interface chatbot tingkat tinggi.

    Dipakai oleh Streamlit UI maupun CLI. Stateless terhadap chat history
    (Streamlit yang menyimpan), tapi MEMILIKI state untuk follow-up tahun libur.
    """

    def __init__(
        self,
        retriever: EnsembleRetriever,
        llm: LLMBackend,
        memory_size: int = 6,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory_size = memory_size
        self._data_dir = data_dir
        self.libur_state = LiburState()

    # -- helpers ----------------------------------------------------------

    def _retrieve_for_libur(self) -> str:
        """Ambil teks kalender lengkap untuk dianalisis regex."""
        docs = self._retriever.invoke(
            "kategori: Perkuliahan dan Pembelajaran Pelaksanaan Perkuliahan dan Praktikum"
        )
        return "\n".join(d.page_content for d in docs)

    def _build_messages(
        self,
        user_input: str,
        history: List[Mapping[str, str]],
    ) -> tuple[List[Mapping[str, str]], List[Document]]:
        """Susun prompt: system + history (truncated) + user-with-context."""
        docs = self._retriever.invoke(user_input)
        context = format_context(docs)
        tanggal_hari_ini = get_tanggal_indo()
        user_with_ctx = build_user_message_with_context(
            user_input, context, tanggal_hari_ini
        )

        truncated_history = list(history)[-self._memory_size :]
        messages: List[Mapping[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(truncated_history)
        messages.append({"role": "user", "content": user_with_ctx})
        return messages, docs

    # -- public API -------------------------------------------------------

    def ask(
        self,
        user_input: str,
        history: Optional[List[Mapping[str, str]]] = None,
    ) -> ChatbotResponse:
        """Proses 1 pertanyaan & kembalikan jawaban + sumber.

        Args:
            user_input: pesan dari user
            history: list of {role, content} pesan sebelumnya (tanpa system & tanpa
                pesan user saat ini). Yang paling lama = paling depan.
        """
        if history is None:
            history = []
        teks_user = user_input.lower()

        # === Branch 0: Pertanyaan 'list semua dosen' ===
        # Top-k retriever (8 chunks) tidak cukup untuk semua 17 dosen, sehingga
        # LLM cenderung halusinasi. Bypass langsung ke daftar dari file.
        if self._data_dir is not None and deteksi_pertanyaan_list_dosen(teks_user):
            names = list_all_dosen(self._data_dir) or []
            return ChatbotResponse(
                answer=format_daftar_dosen(names),
                sources=["dosen_rag_narasi.txt"] if names else [],
                used_special_logic=True,
            )

        # === Branch 1: Sedang menunggu jawaban tahun akademik ===
        if self.libur_state.menunggu_jawaban:
            tahun = deteksi_tahun_akademik(teks_user)
            if tahun:
                sem_awal = self.libur_state.sem_awal
                sem_tujuan = self.libur_state.sem_tujuan
                self.libur_state.reset()
                teks_kal = self._retrieve_for_libur()
                jawaban = hitung_hari_libur(teks_kal, sem_awal, sem_tujuan, tahun)
                return ChatbotResponse(
                    answer=jawaban,
                    sources=["kalender_akademik_rag.txt", "kalender_akademik_rag2627.txt"],
                    used_special_logic=True,
                )
            # User berubah topik → reset & lanjut ke alur normal
            self.libur_state.reset()

        # === Branch 2: Pertanyaan awal tentang hari libur ===
        deteksi = deteksi_pertanyaan_libur(teks_user)
        if deteksi:
            sem_awal, sem_tujuan = deteksi
            tahun = deteksi_tahun_akademik(teks_user)
            if tahun is None:
                self.libur_state.menunggu_jawaban = True
                self.libur_state.sem_awal = sem_awal
                self.libur_state.sem_tujuan = sem_tujuan
                return ChatbotResponse(
                    answer=f"Semester {sem_awal} tahun 2025/2026 atau 2026/2027?",
                    sources=[],
                    used_special_logic=True,
                )
            teks_kal = self._retrieve_for_libur()
            jawaban = hitung_hari_libur(teks_kal, sem_awal, sem_tujuan, tahun)
            return ChatbotResponse(
                answer=jawaban,
                sources=["kalender_akademik_rag.txt", "kalender_akademik_rag2627.txt"],
                used_special_logic=True,
            )

        # === Branch 3: Alur normal RAG + LLM ===
        messages, docs = self._build_messages(user_input, history)
        answer = self._llm.generate(messages)
        return ChatbotResponse(
            answer=answer,
            sources=unique_sources(docs),
            used_special_logic=False,
        )
