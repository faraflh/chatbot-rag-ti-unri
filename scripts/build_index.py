"""One-time script untuk membangun ChromaDB index dari folder data/.

Berguna untuk:
- Pre-build index saat deploy ke Hugging Face Spaces (lebih cepat saat boot).
- Build ulang setelah update dokumen.

Cara pakai:
    python scripts/build_index.py
    python scripts/build_index.py --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Tambahkan project root ke sys.path supaya `from app...` jalan
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import load_settings  # noqa: E402
from app.rag_pipeline import (  # noqa: E402
    build_embeddings,
    build_or_load_vectorstore,
    load_all_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index untuk RAG.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild meskipun chroma_db sudah ada.",
    )
    args = parser.parse_args()

    settings = load_settings()
    print(f"Data dir   : {settings.data_dir}")
    print(f"Chroma dir : {settings.chroma_dir}")
    print(f"Collection : {settings.collection_name}")
    print(f"Embedding  : {settings.embedding_model}")
    print()

    print("Memuat dokumen...")
    documents = load_all_documents(settings.data_dir)
    if not documents:
        sys.exit("Tidak ada dokumen termuat. Periksa folder data/.")

    sizes = [len(d.page_content) for d in documents]
    print(f"\nTotal chunks: {len(documents)}")
    print(
        f"Ukuran chunk: min={min(sizes)}, max={max(sizes)}, "
        f"avg={sum(sizes) // len(sizes)}"
    )

    if args.rebuild and settings.chroma_dir.exists():
        import shutil

        print(f"\nMenghapus index lama di {settings.chroma_dir}...")
        shutil.rmtree(settings.chroma_dir)

    print("\nMemuat embeddings...")
    embeddings = build_embeddings(settings.embedding_model)

    print("\nMembangun vector store...")
    vectorstore = build_or_load_vectorstore(
        documents=documents,
        embeddings=embeddings,
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
        force_rebuild=args.rebuild,
    )

    print(
        f"\nSelesai. Index siap dipakai di {settings.chroma_dir} "
        f"({len(documents)} chunks)."
    )
    _ = vectorstore  # silence linter


if __name__ == "__main__":
    main()
