"""Konfigurasi terpusat dari environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    llm_backend: str
    data_dir: Path
    chroma_dir: Path
    collection_name: str
    embedding_model: str

    hf_model_id: str
    hf_load_in_4bit: bool
    hf_max_new_tokens: int
    hf_temperature: float

    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_max_tokens: int
    openai_temperature: float


def load_settings() -> Settings:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / os.getenv("DATA_DIR", "data")
    chroma_dir = project_root / os.getenv("CHROMA_DIR", "chroma_db")

    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "hf_local").strip().lower(),
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        collection_name=os.getenv("COLLECTION_NAME", "akademik_ti"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "LazarusNLP/all-indo-e5-small-v4"
        ),
        hf_model_id=os.getenv("HF_MODEL_ID", "SeaLLMs/SeaLLM-7B-v2"),
        hf_load_in_4bit=_bool_env("HF_LOAD_IN_4BIT", True),
        hf_max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "300")),
        hf_temperature=float(os.getenv("HF_TEMPERATURE", "0.1")),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
        openai_model=os.getenv("OPENAI_MODEL", "local-model"),
        openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "300")),
        openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.1")),
    )
