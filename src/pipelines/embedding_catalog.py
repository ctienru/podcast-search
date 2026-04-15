"""Embedding catalog — authoritative MODEL_MAP for the search pipeline.

Phase 2a moves `MODEL_MAP` out of `src.embedding.backend` so that drift
detection, cache validation, and the rebuild primitive can consume it without
pulling in the heavy `sentence_transformers` dependency chain.

Contract (Phase 2a §3.8):
- This module is pure configuration; no imports from `src.embedding`,
  `src.pipelines.embed_episodes`, or anything runtime-heavy.
- `MODEL_MAP` keys are the language keys understood by `LocalEmbeddingBackend`
  (`"zh"`, `"en"`). Values are HuggingFace model identifiers.
- `src.embedding.backend` re-exports `MODEL_MAP` from here for backward
  compatibility with existing callers (see `backend.py`).
"""

from __future__ import annotations

MODEL_MAP: dict[str, str] = {
    "zh": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim, zh-tw + zh-cn
    "en": "paraphrase-multilingual-MiniLM-L12-v2",  # 384 dim
}
