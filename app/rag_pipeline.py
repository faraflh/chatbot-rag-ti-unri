"""Pipeline RAG: dokumen loading, chunking, embeddings, ChromaDB, hybrid retrieval.

Diadaptasi dari notebook rag_buat_ui.ipynb dengan strategi chunking yang sama:
- File `dosen_*` → chunk_size=400, separator hanya newline (1 dosen = 1 chunk)
- File `kalender_*` → chunk_size=1000, overlap=200
- File markdown SOP/Pedoman/Kurikulum → MarkdownHeaderTextSplitter + recursive split

Hybrid retriever: BM25 (Sastrawi tokenizer) + Vector (LazarusNLP indo-e5).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

_STOP_WORDS = set(StopWordRemoverFactory().get_stop_words())


def id_tokenize(text: str) -> List[str]:
    """Tokenizer Bahasa Indonesia untuk BM25 (lowercase + buang stopword)."""
    text = text.lower()
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


class E5Embeddings(HuggingFaceEmbeddings):
    """LazarusNLP indo-e5 perlu prefix `passage:` (doc) & `query:` (kueri)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return super().embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return super().embed_query(f"query: {text}")


def load_txt_data(file_path: Path) -> List[Document]:
    """Load file .txt — beda strategi untuk profil dosen vs kalender akademik."""
    file_name = file_path.name
    content = file_path.read_text(encoding="utf-8")

    doc_type = "kalender_akademik" if "kalender" in file_name.lower() else "profil_dosen"

    if doc_type == "profil_dosen":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=0,
            separators=["\n"],
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    chunks = splitter.split_text(content)
    documents: List[Document] = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        documents.append(
            Document(
                page_content=chunk.strip(),
                metadata={
                    "source": file_name,
                    "type": doc_type,
                    "chunk_id": i,
                },
            )
        )
    return documents


def load_markdown_data(file_path: Path) -> List[Document]:
    """Load file markdown — split per heading lalu recursive char split."""
    file_name = file_path.name
    content = file_path.read_text(encoding="utf-8")

    name_lower = file_name.lower()
    if "kurikulum" in name_lower:
        doc_type = "kurikulum"
    elif "sop" in name_lower or "pedoman" in name_lower:
        doc_type = "pedoman_akademik"
    else:
        doc_type = "informasi_umum"

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = md_splitter.split_text(content)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    final_chunks = char_splitter.split_documents(md_header_splits)

    for i, chunk in enumerate(final_chunks):
        chunk.metadata["source"] = file_name
        chunk.metadata["type"] = doc_type
        chunk.metadata["chunk_id"] = i
    return list(final_chunks)


# Daftar file yang akan diindeks (urutan = urutan loading di notebook asli).
TXT_FILES = [
    "dosen_rag_narasi.txt",
    "kalender_akademik_rag.txt",
    "kalender_akademik_rag2627.txt",
]
MD_FILES = [
    "sop_kp2.md",
    "sop_skripsi.md",
    "Pedoman_Skripsi_Sangat_Bersih.md",
    "kurikulum_rag_lengkap.md",
    "informasi_umum.md",
]


def load_all_documents(data_dir: Path) -> List[Document]:
    """Load semua dokumen dari `data_dir`."""
    all_docs: List[Document] = []
    for fname in TXT_FILES:
        path = data_dir / fname
        if not path.exists():
            print(f"   [WARN] File tidak ditemukan: {path}")
            continue
        docs = load_txt_data(path)
        print(f"   Loaded {len(docs)} chunks dari {fname}")
        all_docs.extend(docs)

    for fname in MD_FILES:
        path = data_dir / fname
        if not path.exists():
            print(f"   [WARN] File tidak ditemukan: {path}")
            continue
        docs = load_markdown_data(path)
        print(f"   Loaded {len(docs)} chunks dari {fname}")
        all_docs.extend(docs)

    return all_docs


def build_embeddings(embedding_model: str) -> E5Embeddings:
    """Bangun embeddings (auto pakai CUDA jika tersedia)."""
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    return E5Embeddings(
        model_name=embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_or_load_vectorstore(
    documents: List[Document],
    embeddings: E5Embeddings,
    persist_directory: Path,
    collection_name: str,
    force_rebuild: bool = False,
) -> Chroma:
    """Bangun ChromaDB; reuse jika folder sudah ada (kecuali force_rebuild)."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    chroma_files_exist = any(persist_directory.glob("chroma.sqlite3*"))

    if chroma_files_exist and not force_rebuild:
        print(f"   Memuat ChromaDB dari {persist_directory} (cache)")
        return Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    print(f"   Membangun ChromaDB di {persist_directory}")
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_directory),
        collection_name=collection_name,
    )


def build_hybrid_retriever(
    documents: List[Document],
    vectorstore: Chroma,
    bm25_k: int = 5,
    vector_k: int = 3,
    weights: tuple[float, float] = (0.3, 0.7),
) -> EnsembleRetriever:
    """Hybrid retriever: BM25 + Vector dengan weight sesuai notebook asli."""
    bm25_retriever = BM25Retriever.from_documents(
        documents, preprocess_func=id_tokenize
    )
    bm25_retriever.k = bm25_k

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": vector_k},
    )

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=list(weights),
    )


def format_context(docs: List[Document]) -> str:
    """Format konteks RAG sesuai notebook asli."""
    parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "-")
        tipe = doc.metadata.get("type", "Umum")
        parts.append(
            f"[Dokumen {i} | Sumber: {source} | Tipe: {tipe}]\n{doc.page_content}"
        )
    return "\n\n".join(parts)


def unique_sources(docs: List[Document]) -> List[str]:
    """Daftar sumber unik dari hasil retrieval."""
    seen: List[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "-")
        if src not in seen:
            seen.append(src)
    return seen
