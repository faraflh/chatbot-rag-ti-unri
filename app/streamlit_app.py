"""Streamlit UI untuk Chatbot RAG Akademik TI UNRI.

Jalankan dengan:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Pastikan import berfungsi baik dari root maupun saat dipanggil langsung
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.chatbot import Chatbot  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.llm_backend import build_llm_backend  # noqa: E402
from app.rag_pipeline import (  # noqa: E402
    build_embeddings,
    build_hybrid_retriever,
    build_or_load_vectorstore,
    load_all_documents,
)


# ---------- Page config (harus pertama) ----------

st.set_page_config(
    page_title="Chatbot Akademik TI UNRI",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------- Cache pipeline ----------


@st.cache_resource(show_spinner="Memuat embeddings & vector store...")
def _bootstrap_retriever(settings: Settings):
    """Cache: embeddings + ChromaDB + hybrid retriever (sekali per session)."""
    documents = load_all_documents(settings.data_dir)
    if not documents:
        raise RuntimeError(
            f"Tidak ada dokumen termuat dari {settings.data_dir}. "
            "Pastikan folder data/ berisi file-file RAG."
        )
    embeddings = build_embeddings(settings.embedding_model)
    vectorstore = build_or_load_vectorstore(
        documents=documents,
        embeddings=embeddings,
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    retriever = build_hybrid_retriever(documents=documents, vectorstore=vectorstore)
    return retriever, len(documents)


@st.cache_resource(show_spinner="Memuat LLM backend...")
def _bootstrap_llm(settings: Settings):
    return build_llm_backend(settings)


def _get_chatbot(settings: Settings) -> Chatbot:
    """Bangun chatbot — retriever & LLM di-cache, tapi instance Chatbot
    tetap di-recreate agar libur_state ter-reset di awal session."""
    if "chatbot" in st.session_state:
        return st.session_state.chatbot

    retriever, n_docs = _bootstrap_retriever(settings)
    st.session_state.n_docs = n_docs
    llm = _bootstrap_llm(settings)
    chatbot = Chatbot(retriever=retriever, llm=llm, data_dir=settings.data_dir)
    st.session_state.chatbot = chatbot
    return chatbot


# ---------- Sidebar ----------


def _render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.title("🎓 Chatbot Akademik")
        st.caption("Program Studi Teknik Informatika — Universitas Riau")

        st.divider()
        st.subheader("Konfigurasi")
        st.write(f"**LLM Backend**: `{settings.llm_backend}`")
        if settings.llm_backend == "hf_local":
            st.write(f"**Model**: `{settings.hf_model_id}`")
        else:
            st.write(f"**Endpoint**: `{settings.openai_base_url}`")
            st.write(f"**Model**: `{settings.openai_model}`")
        st.write(f"**Embedding**: `{settings.embedding_model}`")
        if "n_docs" in st.session_state:
            st.write(f"**Total chunks terindeks**: {st.session_state.n_docs}")

        st.divider()
        st.subheader("Topik yang bisa ditanyakan")
        st.markdown(
            """
- 👨‍🏫 Profil dosen TI UNRI
- 📅 Kalender akademik (TA 25/26 & 26/27)
- 📖 Kurikulum 2018 / 2025 + SKS
- 📝 SOP Kerja Praktik
- 🎓 SOP Skripsi & pedoman penulisan
- 📰 Informasi umum prodi
            """
        )

        st.divider()
        if st.button("🔄 Reset percakapan", use_container_width=True):
            st.session_state.messages = []
            if "chatbot" in st.session_state:
                st.session_state.chatbot.libur_state.reset()
            st.rerun()


# ---------- Main ----------


def main() -> None:
    settings = load_settings()
    _render_sidebar(settings)

    st.title("Chatbot Akademik TI UNRI")
    st.caption(
        "Tanyakan apa saja seputar akademik Prodi Teknik Informatika Universitas Riau."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sapaan awal
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                "Halo! Saya **Asisten Akademik TI UNRI**. Saya bisa membantu menjawab "
                "pertanyaan tentang dosen, kalender akademik, kurikulum, SOP Kerja "
                "Praktik, SOP Skripsi, dan informasi umum prodi.\n\n"
                "Silakan ketik pertanyaan Anda di bawah."
            )

    # Tampilkan history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Sumber dokumen"):
                    for src in msg["sources"]:
                        st.markdown(f"- `{src}`")

    # Input user
    user_input = st.chat_input("Ketik pertanyaan Anda di sini...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Bangun chatbot lazy (supaya error config baru muncul saat user mulai chat)
    try:
        chatbot = _get_chatbot(settings)
    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(
                f"Gagal menginisialisasi chatbot: {exc}\n\n"
                "Periksa konfigurasi `.env` dan dependensi terinstall."
            )
        return

    # History tanpa pesan user terakhir (sudah masuk ke prompt secara terpisah)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban..."):
            try:
                response = chatbot.ask(user_input, history=history)
            except Exception as exc:
                err = f"Maaf, terjadi kesalahan: {exc}"
                st.error(err)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err, "sources": []}
                )
                return

        st.markdown(response.answer)
        if response.sources:
            with st.expander("📚 Sumber dokumen"):
                for src in response.sources:
                    st.markdown(f"- `{src}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": response.sources,
        }
    )


if __name__ == "__main__":
    main()
