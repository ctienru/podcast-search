"""Phase 2a text primitive contract tests.

Covers:
- Basic shape: TextChunk + prepare_chunks_for_show produces 1:1 chunks
- Deterministic: same input → same output (incl. order)
- Identity dependency boundary: only embedding_version may drive chunk
  behavior; model_name / embedding_dimensions changes must not affect output
- Filtering: records missing episode_id or text are dropped silently
  (matches pre-primitive behavior that would raise or skip)
"""

from __future__ import annotations

from src.pipelines.embedding_identity import EmbeddingIdentity
from src.pipelines.embedding_text import TextChunk, prepare_chunks_for_show


def _identity(
    *,
    model: str = "paraphrase-multilingual-MiniLM-L12-v2",
    version: str = "text-v1",
    dims: int = 384,
) -> EmbeddingIdentity:
    return EmbeddingIdentity(
        model_name=model, embedding_version=version, embedding_dimensions=dims,
    )


def _mk(episode_id: str, text: str) -> dict:
    return {"episode_id": episode_id, "embedding_input": {"text": text}}


# ── Basic shape ─────────────────────────────────────────────────────────────

def test_returns_one_chunk_per_episode_with_chunk_index_zero() -> None:
    """Phase 2a chunking policy: 1:1 mapping, chunk_index=0."""
    chunks = prepare_chunks_for_show(
        show_id="show:1",
        episode_inputs=[_mk("ep:1", "hello"), _mk("ep:2", "world")],
        identity=_identity(),
    )
    assert len(chunks) == 2
    for c in chunks:
        assert isinstance(c, TextChunk)
        assert c.chunk_index == 0


def test_text_is_taken_verbatim_from_embedding_input() -> None:
    chunks = prepare_chunks_for_show(
        show_id="show:1",
        episode_inputs=[_mk("ep:1", "  hello  world\n")],
        identity=_identity(),
    )
    assert chunks[0].text == "  hello  world\n"


def test_empty_input_yields_empty_list() -> None:
    assert prepare_chunks_for_show(
        show_id="show:1", episode_inputs=[], identity=_identity(),
    ) == []


# ── Determinism (TP2) ───────────────────────────────────────────────────────

def test_chunking_primitive_is_deterministic() -> None:
    """Two calls with equivalent inputs produce identical outputs.

    Determinism is how embed_episodes and show_rebuild agree on what
    vectors a cache file should contain. Include `chunk_index`, `text`,
    and `episode_id` in the equality check — order matters.
    """
    inputs = [_mk("ep:b", "bee"), _mk("ep:a", "ay"), _mk("ep:c", "cee")]

    result_1 = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(),
    )
    result_2 = prepare_chunks_for_show(
        show_id="s", episode_inputs=list(reversed(inputs)), identity=_identity(),
    )

    assert result_1 == result_2
    # Specifically: ordered ascending by episode_id
    assert [c.episode_id for c in result_1] == ["ep:a", "ep:b", "ep:c"]


# ── Identity dependency boundary (§3.8) ─────────────────────────────────────

def test_changing_model_name_does_not_affect_chunks() -> None:
    """`identity.model_name` is a runtime concern; must not drive chunking."""
    inputs = [_mk("ep:1", "hello")]
    chunks_a = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(model="modelA"),
    )
    chunks_b = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(model="modelB"),
    )
    assert chunks_a == chunks_b


def test_changing_embedding_dimensions_does_not_affect_chunks() -> None:
    """`identity.embedding_dimensions` is a validation concern; not a chunking lever."""
    inputs = [_mk("ep:1", "hello")]
    chunks_384 = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(dims=384),
    )
    chunks_768 = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(dims=768),
    )
    assert chunks_384 == chunks_768


# ── Filtering ───────────────────────────────────────────────────────────────

def test_drops_records_with_missing_episode_id() -> None:
    inputs = [
        _mk("ep:1", "hello"),
        {"embedding_input": {"text": "no id here"}},  # missing episode_id
    ]
    chunks = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(),
    )
    assert len(chunks) == 1
    assert chunks[0].episode_id == "ep:1"


def test_drops_records_with_missing_or_empty_text() -> None:
    inputs = [
        _mk("ep:1", "hello"),
        {"episode_id": "ep:2"},  # no embedding_input
        {"episode_id": "ep:3", "embedding_input": {}},  # no text
        {"episode_id": "ep:4", "embedding_input": {"text": ""}},  # empty text
    ]
    chunks = prepare_chunks_for_show(
        show_id="s", episode_inputs=inputs, identity=_identity(),
    )
    assert [c.episode_id for c in chunks] == ["ep:1"]
