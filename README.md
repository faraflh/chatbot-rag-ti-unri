# Chatbot RAG Akademik — Teknik Informatika UNRI

Chatbot berbahasa Indonesia untuk Program Studi Teknik Informatika Universitas Riau,
menggunakan **Retrieval-Augmented Generation (RAG)** dengan Streamlit UI.

> Berbasis notebook `rag_buat_ui.ipynb` (Google Colab) yang dirombak menjadi aplikasi web
> dengan backend LLM yang **dapat diganti via konfigurasi** (lokal SeaLLM untuk Colab/laptop
> ber-GPU, atau API OpenAI-compatible untuk LM Studio / Ollama / Groq / dll.).

## ✨ Fitur

- 🤖 **Chat UI** modern berbasis Streamlit (chat bubbles + history + sumber dokumen).
- 🇮🇩 **Embedding khusus Bahasa Indonesia**: `LazarusNLP/all-indo-e5-small-v4`.
- 🔍 **Hybrid retrieval**: BM25 (Sastrawi tokenizer) + vector similarity (ChromaDB), bobot 0.3/0.7.
- 📅 **Logic khusus kalender**: hitung hari libur antar-semester berdasarkan tanggal nyata.
- 🎯 **System prompt akademik resmi** dengan 8 aturan WAJIB (anti-halusinasi, format dosen, kurikulum, SOP, dll.).
- 🔌 **LLM backend modular**:
  - `hf_local` — SeaLLM-7B-v2 lokal via HuggingFace transformers + 4-bit quantization (butuh GPU).
  - `openai_compat` — endpoint OpenAI-compatible (LM Studio, Ollama, Groq, OpenRouter, Together AI, dll.).

## 📂 Struktur Project

```
.
├── app/
│   ├── chatbot.py          # Orchestrator (RAG + kalender logic + LLM)
│   ├── config.py           # Konfigurasi dari .env
│   ├── kalender_logic.py   # Hitung hari libur antar-semester
│   ├── llm_backend.py      # Abstraksi LLM (hf_local / openai_compat)
│   ├── prompts.py          # SYSTEM_PROMPT akademik
│   ├── rag_pipeline.py     # Loader, chunking, embeddings, ChromaDB, hybrid retrieval
│   └── streamlit_app.py    # UI Streamlit
├── data/                   # Dokumen RAG (.txt & .md)
├── scripts/
│   └── build_index.py      # Pre-build vector store
├── .streamlit/config.toml
├── app.py                  # Entrypoint (HF Spaces / Streamlit Cloud)
├── Dockerfile              # Untuk HF Spaces (Docker SDK)
├── requirements.txt        # Dependensi inti (CPU-friendly)
└── requirements-local-llm.txt  # Tambahan untuk mode SeaLLM lokal
```

## 🚀 Cara Menjalankan

### A. Lokal (laptop) — Mode `openai_compat` dengan LM Studio / Ollama

Cocok untuk laptop **tanpa GPU**. LLM dijalankan oleh app eksternal (LM Studio atau Ollama)
yang sudah meng-handle quantization & efisiensi CPU.

```bash
# 1. Clone & setup
git clone https://github.com/faraflh/chatbot-rag-ti-unri.git
cd chatbot-rag-ti-unri
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Konfigurasi
cp .env.example .env
# Edit .env:
#   LLM_BACKEND=openai_compat
#   OPENAI_BASE_URL=http://localhost:1234/v1   (LM Studio default)
#   OPENAI_API_KEY=lm-studio
#   OPENAI_MODEL=<nama-model-yang-Anda-load-di-LM-Studio>

# 3. Jalankan LM Studio / Ollama dengan model pilihan (mis. SeaLLM-7B-v2 GGUF)
#    LM Studio → tab "Local Server" → Start Server
#    Ollama   → ollama serve   (default port 11434)

# 4. Build index (sekali; cache di chroma_db/)
python scripts/build_index.py

# 5. Jalankan UI
streamlit run app.py
# Buka http://localhost:8501
```

> **Tip Ollama**: untuk endpoint OpenAI-compatible Ollama, set `OPENAI_BASE_URL=http://localhost:11434/v1`
> dan `OPENAI_MODEL=<nama-model-ollama>` (mis. `llama3.1:8b`, `qwen2.5:7b`).

### B. Google Colab — Mode `hf_local` dengan SeaLLM (sesuai notebook asli)

Cocok jika Anda **tetap pakai SeaLLM-7B-v2** dengan GPU T4 gratis di Colab.

```python
# Cell 1 — Clone & setup
!git clone https://github.com/faraflh/chatbot-rag-ti-unri.git
%cd chatbot-rag-ti-unri
!pip install -r requirements-local-llm.txt
!pip install pyngrok        # Untuk expose port keluar Colab

# Cell 2 — Konfigurasi
import os
os.environ["LLM_BACKEND"] = "hf_local"
os.environ["HF_MODEL_ID"] = "SeaLLMs/SeaLLM-7B-v2"
os.environ["HF_LOAD_IN_4BIT"] = "true"

# Cell 3 — Build index (sekali; ~30 detik)
!python scripts/build_index.py

# Cell 4 — Jalankan Streamlit + ngrok tunnel
from pyngrok import ngrok
import threading, subprocess, time

ngrok.set_auth_token("YOUR_NGROK_TOKEN")   # Dari https://dashboard.ngrok.com/

def run():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501",
                    "--server.headless", "true"])

threading.Thread(target=run, daemon=True).start()
time.sleep(8)
public_url = ngrok.connect(8501)
print("Chatbot tersedia di:", public_url)
```

### C. Hugging Face Spaces (deployment publik)

> **Penting**: HF Spaces **free tier = CPU only**. Mode `hf_local` SeaLLM-7B-v2 4-bit
> **TIDAK akan jalan** di tier gratis karena `bitsandbytes` membutuhkan CUDA.
> Untuk deployment publik gratis, gunakan **mode `openai_compat`** dengan provider API gratis
> seperti **Groq** atau **Google Gemini**.

#### Langkah deployment:

1. **Buat Space baru** di https://huggingface.co/new-space
   - Owner: akun Anda
   - SDK: **Docker** (gunakan Dockerfile di repo) ATAU **Streamlit** (auto-deteksi `app.py`)
   - Hardware: **CPU basic — Free**

2. **Push repo ini** ke remote Space:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/<space-name>
   git push space main
   ```

3. **Set Secrets** di Settings Space:
   - `LLM_BACKEND` = `openai_compat`
   - `OPENAI_BASE_URL` = `https://api.groq.com/openai/v1` (untuk Groq)
   - `OPENAI_API_KEY` = `gsk_...` (dari https://console.groq.com/keys)
   - `OPENAI_MODEL` = `llama-3.1-8b-instant`

4. **Tunggu build & deploy**. Saat boot pertama, ChromaDB akan terbangun otomatis.

## 📚 Data Sources

Folder `data/` berisi 8 dokumen RAG (1.832 baris total):

| File | Tipe | Isi |
|---|---|---|
| `dosen_rag_narasi.txt` | profil_dosen | 17 dosen TI UNRI (NIP, jabatan, kepakaran, email) |
| `kalender_akademik_rag.txt` | kalender_akademik | TA 2025/2026 |
| `kalender_akademik_rag2627.txt` | kalender_akademik | TA 2026/2027 |
| `informasi_umum.md` | informasi_umum | Sejarah, visi-misi, fasilitas |
| `kurikulum_rag_lengkap.md` | kurikulum | Kurikulum 2018 + 2025 |
| `Pedoman_Skripsi_Sangat_Bersih.md` | pedoman_akademik | Pedoman penulisan skripsi |
| `sop_kp2.md` | pedoman_akademik | SOP Kerja Praktik |
| `sop_skripsi.md` | pedoman_akademik | SOP Skripsi |

## 🧠 Arsitektur RAG

```
User question
    │
    ▼
┌──────────────────────────────────────────────┐
│ Deteksi: pertanyaan tentang "hari libur"?    │
└──────────────┬───────────────────────────────┘
               │
       ┌───────┴───────┐
      yes              no
       │                │
       ▼                ▼
   kalender_logic   Hybrid Retriever
   (regex match)    BM25 (k=5) + Vector (k=3)
       │                │ weights: 0.3 / 0.7
       │                ▼
       │           Top-N relevant chunks
       │                │
       │                ▼
       │           Prompt = SYSTEM + history + (CONTEXT + WAKTU + QUERY)
       │                │
       │                ▼
       │           LLM (hf_local SeaLLM ATAU openai_compat)
       │                │
       └───────►◄───────┘
               ▼
        Jawaban + sumber dokumen
```

## ⚙️ Konfigurasi Environment

Lihat `.env.example` untuk semua variabel.

| Variabel | Default | Keterangan |
|---|---|---|
| `LLM_BACKEND` | `hf_local` | `hf_local` atau `openai_compat` |
| `HF_MODEL_ID` | `SeaLLMs/SeaLLM-7B-v2` | Model HuggingFace untuk mode lokal |
| `HF_LOAD_IN_4BIT` | `true` | Quantization 4-bit (butuh CUDA) |
| `OPENAI_BASE_URL` | `http://localhost:1234/v1` | Endpoint OpenAI-compatible |
| `OPENAI_API_KEY` | `lm-studio` | API key (bebas untuk LM Studio lokal) |
| `OPENAI_MODEL` | `local-model` | Nama model |
| `EMBEDDING_MODEL` | `LazarusNLP/all-indo-e5-small-v4` | Embedding model |
| `CHROMA_DIR` | `chroma_db` | Folder vector store |
| `COLLECTION_NAME` | `akademik_ti` | Nama collection ChromaDB |

## 🔧 Penggunaan Lanjutan

### Build ulang index setelah update dokumen

```bash
python scripts/build_index.py --rebuild
```

### Cek apakah ChromaDB berfungsi (CLI test)

```python
from app.config import load_settings
from app.rag_pipeline import build_embeddings, build_or_load_vectorstore, load_all_documents, build_hybrid_retriever

s = load_settings()
docs = load_all_documents(s.data_dir)
emb = build_embeddings(s.embedding_model)
vs = build_or_load_vectorstore(docs, emb, s.chroma_dir, s.collection_name)
ret = build_hybrid_retriever(docs, vs)

results = ret.invoke("siapa koordinator program studi TI UNRI?")
for d in results:
    print(d.metadata.get("source"), "→", d.page_content[:100])
```

## 📜 Catatan tentang Hosting Gratis

| Platform | Apakah Cocok? | Catatan |
|---|---|---|
| **Vercel** | ❌ | Serverless, tidak bisa untuk model ML besar + ChromaDB persisten |
| **GitHub Pages** | ❌ | Static only, tidak ada Python backend |
| **Streamlit Community Cloud** | ✅ | Free, 1 GB RAM, CPU only — pakai mode `openai_compat` |
| **Hugging Face Spaces** | ✅ | Free, 16 GB RAM, CPU only — pakai mode `openai_compat` |
| **Render / Railway / Fly.io** | ⚠️ | Ada batasan free tier / spin-down |

Untuk LLM gratis lewat API, lihat:
- **Groq**: https://console.groq.com (Llama 3.1 / 3.3 — sangat cepat)
- **Google Gemini**: https://aistudio.google.com (free tier 15 req/min)
- **OpenRouter**: https://openrouter.ai (beberapa model free)

## 📝 Lisensi

MIT — silakan dipakai & dimodifikasi.
