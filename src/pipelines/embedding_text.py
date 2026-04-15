"""Embedding text-assembly + chunking primitive — Phase 2a.

This is the sole owner of "what text gets embedded for a given show".
Pipelines that embed content (`embed_episodes`, upcoming `show_rebuild`)
MUST route through `prepare_chunks_for_show` — they are forbidden from
implementing their own text assembly or chunking. Without this primitive
the two callers could silently diverge and the cache identity contract
would no longer be trustworthy (two runs of "the same" version producing
different vectors for the same show).

Phase 2a chunking policy is a 1:1 mapping: one chunk per episode,
`chunk_index=0`, text taken verbatim from the upstream embedding_input
artifact. Future versions may split long episodes — when that happens,
the chunking change MUST bump `EMBEDDING_TEXT_VERSION` so cache identity
detects it.

Identity dependency boundary:
- Only `identity.embedding_version` is allowed to drive chunking behavior.
- `identity.model_name` and `identity.embedding_dimensions` are
  runtime / validation concerns and must not influence chunk generation.
- A test pins this (same show + episodes, different model or dims →
  identical chunks; different embedding_version → may differ).

Pure compute: no filesystem, no network, no environment reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.pipelines.embedding_identity import EmbeddingIdentity


@dataclass(frozen=True)
class TextChunk:
    """One unit of text to embed.

    Phase 2a keeps a 1:1 episode-to-chunk policy (chunk_index always 0).
    The field exists so that future text-version bumps can split episodes
    without rewriting the primitive's return shape.
    """

    chunk_index: int
    text: str
    episode_id: str


def prepare_chunks_for_show(
    *,
    show_id: str,
    episode_inputs: Sequence[Mapping[str, Any]],
    identity: EmbeddingIdentity,
) -> list[TextChunk]:
    """Return the ordered chunks to embed for a given show.

    Args:
        show_id: The show being processed (currently used only for future
            chunking policies that depend on show-level fields; Phase 2a
            does not read it but keeps it in the signature so callers
            always pass an unambiguous show scope).
        episode_inputs: Pre-assembled embedding-input records for the
            episodes belonging to `show_id`. Each element must expose:
                - `"episode_id"`: non-empty string
                - `"embedding_input"`: dict with a `"text"` string
            Produced upstream by `prepare_embedding_input.py`. Records
            missing either field are skipped (matches the behavior of
            the pre-primitive path).
        identity: The expected embedding identity. Phase 2a chunking
            policy depends only on `identity.embedding_version`;
            `model_name` and `embedding_dimensions` are intentionally
            unused here and their change must not affect output.

    Returns:
        A list of `TextChunk` in **deterministic order** (ascending
        `episode_id`), so two calls with equivalent inputs always produce
        byte-identical chunk sequences. Determinism lets `show_rebuild`
        and `embed_episodes` agree on the cache payload they produce.
    """
    _ = show_id, identity  # reserved for future chunking policies

    chunks: list[TextChunk] = []
    for record in sorted(episode_inputs, key=lambda r: r.get("episode_id", "")):
        episode_id = record.get("episode_id")
        embedding_input = record.get("embedding_input") or {}
        text = embedding_input.get("text") if isinstance(embedding_input, Mapping) else None

        if not episode_id or not text:
            continue

        chunks.append(TextChunk(chunk_index=0, text=text, episode_id=episode_id))

    return chunks
