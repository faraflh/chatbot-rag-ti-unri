"""Abstraksi backend LLM yang dapat diganti via env var.

Dua mode didukung:
- ``hf_local``: SeaLLM (atau model HF lain) via transformers + bitsandbytes 4-bit.
                Membutuhkan GPU CUDA.
- ``openai_compat``: API OpenAI-compatible (LM Studio, Ollama, Groq, OpenRouter,
                     Together AI, dll.). Hanya butuh HTTP, jalan di CPU/laptop apa saja.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import List, Mapping

from .config import Settings


class LLMBackend(ABC):
    """Interface untuk semua backend LLM."""

    @abstractmethod
    def generate(self, messages: List[Mapping[str, str]]) -> str:
        """Hasilkan response dari list of {role, content}."""


def _clean_response(text: str) -> str:
    """Bersihkan output LLM dari prefix/duplikasi (porting dari notebook)."""
    text = re.sub(
        r"^(Based on.*?:|Here is.*?:|Berikut.*?jawaban.*?:)\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    for marker in ["Human:", "Pertanyaan:", "Assistant:", "User:", "KONTEKS:", "INSTRUKSI:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    out, prev = [], None
    for ln in text.splitlines():
        if ln.strip() == prev:
            continue
        out.append(ln)
        prev = ln.strip()
    return "\n".join(out).strip()


class HFLocalBackend(LLMBackend):
    """SeaLLM lokal via HuggingFace transformers (butuh GPU)."""

    def __init__(self, settings: Settings):
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                pipeline,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Mode hf_local memerlukan paket: torch, transformers, accelerate, "
                "bitsandbytes. Install dengan: pip install -r requirements-local-llm.txt"
            ) from exc

        model_id = settings.hf_model_id
        print(f"[LLM] Loading {model_id} ...")

        kwargs: dict = {"device_map": "auto", "low_cpu_mem_usage": True}
        if settings.hf_load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "HF_LOAD_IN_4BIT=true butuh CUDA. Set HF_LOAD_IN_4BIT=false "
                    "untuk CPU (akan sangat lambat) atau pakai LLM_BACKEND=openai_compat."
                )
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self._pipe = pipeline(
            "text-generation",
            model=self._model,
            tokenizer=self._tokenizer,
            max_new_tokens=settings.hf_max_new_tokens,
            temperature=settings.hf_temperature,
            do_sample=settings.hf_temperature > 0,
            pad_token_id=self._tokenizer.eos_token_id,
            return_full_text=False,
        )
        print("[LLM] Model SeaLLM siap digunakan.")

    def generate(self, messages: List[Mapping[str, str]]) -> str:
        prompt = self._tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        outputs = self._pipe(prompt)
        text = outputs[0]["generated_text"].strip()
        return _clean_response(text)


class OpenAICompatBackend(LLMBackend):
    """LLM via API OpenAI-compatible (LM Studio, Ollama, Groq, OpenRouter, dll.)."""

    def __init__(self, settings: Settings):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Mode openai_compat memerlukan paket `openai`. "
                "Install dengan: pip install openai"
            ) from exc

        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
        self._model = settings.openai_model
        self._max_tokens = settings.openai_max_tokens
        self._temperature = settings.openai_temperature
        print(
            f"[LLM] OpenAI-compatible client → {settings.openai_base_url} "
            f"(model: {self._model})"
        )

    def generate(self, messages: List[Mapping[str, str]]) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[dict(m) for m in messages],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _clean_response(text)


def build_llm_backend(settings: Settings) -> LLMBackend:
    """Factory: bangun backend sesuai `LLM_BACKEND` env."""
    backend = settings.llm_backend
    if backend == "hf_local":
        return HFLocalBackend(settings)
    if backend == "openai_compat":
        return OpenAICompatBackend(settings)
    raise ValueError(
        f"LLM_BACKEND tidak dikenali: {backend!r}. "
        "Gunakan 'hf_local' atau 'openai_compat'."
    )
