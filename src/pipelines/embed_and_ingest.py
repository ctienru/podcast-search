"""Embed and Ingest Episodes Pipeline

Read embedding_input files, generate embeddings, and write to Elasticsearch.

Data sources:
- embedding_input: data/embedding_input/episodes/*.json (text for embedding)
- cleaned: data/cleaned/episodes/*.json (episode metadata)
- shows: SQLiteStorage (show metadata)

Usage:
    python -m src.pipelines.embed_and_ingest
    python -m src.pipelines.embed_and_ingest --batch-size 128
    python -m src.pipelines.embed_and_ingest --dry-run
    python -m src.pipelines.embed_and_ingest --force-full              # ignore cursor
    python -m src.pipelines.embed_and_ingest --show-id <id>            # SYNC_MODE=single
    python -m src.pipelines.embed_and_ingest --show-ids id1 id2 id3    # process specific shows

Ingest Commit Semantics
(Phase 1 scope: daily incremental path only; NOT a module-wide guarantee)
-------------------------------------------------------------------------
The daily incremental path (`run_incremental` + `EmbedAndIngestPipeline.run()`)
follows a strict commit boundary (spec R1–R7):

- `ingest_cursor.json` is written exactly ONCE, at the end of a fully successful run
- On any partial failure (mixed success/error in `pipeline.run()` stats), cursor is NOT advanced
- `search_sync_state` flush is all-or-nothing: if any error occurred, no row is marked 'synced'
- On an empty candidate set (no shows updated since cursor), BOTH `last_ingest_at` and
  `last_run_at` are frozen — no-op run leaves zero footprint on the cursor
- `search_sync_state` writes use the `environment` value passed to the pipeline's
  constructor; the daily incremental path injects `'local'` (architecture rule §3.1.1).
  The class no longer reads `settings.ES_ENV` directly.
- The daily CLI path (`SYNC_MODE=incremental` or `--force-full`) propagates non-zero
  exit code on any ingest error. `SYNC_MODE=single` / `SYNC_MODE=backfill` paths
  retain pre-Phase-1 exit semantics:
  - `backfill` is a thin wrapper around `run_incremental(force_full=True)`, so it
    inherits this phase's all-or-nothing / cursor-freeze behavior as a side effect,
    and its sync_state writes are now pinned to `environment='local'`
  - `single` (via `upsert_by_show_id`) is NOT covered by Sev0 correctness yet

Re-run safety depends on idempotent ES upsert (same `_id = episode_id`, no
non-deterministic fields). Verified by Step 0 + integration test.

Transaction boundary is owned by the pipeline layer (this module).
`SyncStateRepository` remains a dumb write API; do NOT introduce auto-commit there
or this guarantee breaks.

Errors counter coverage assumption (A1): `stats["errors"]` is treated as a step-level
gate. The `run()` method funnels bulk/transform/flush exceptions into `errors` so
the gate remains honest; if new failure modes are added, they MUST be funnelled too
or the all-or-nothing guarantee is silently broken.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

from elasticsearch.helpers import streaming_bulk
from sqlite_utils import Database
from tqdm import tqdm

from src.config import settings
from src.embedding.backend import EmbeddingBackend, LocalEmbeddingBackend
from src.pipelines.embedding_identity import (
    EmbeddingDimensionContractViolation,
    EmbeddingIdentity,
    resolve_expected_identity,
)
from src.pipelines.embedding_paths import cache_path_for, validate_cache_identity
from src.pipelines.exceptions import CacheMissError
from src.pipelines.show_rebuild import ShowRebuildResult, rebuild_show_cache
from src.search.routing import IndexRoutingStrategy, LanguageSplitRoutingStrategy
from src.services.es_service import ElasticsearchService
from src.storage.base import StorageBase
from src.storage.episode_status import EpisodeStatusRepository
from src.storage.factory import create_storage
from src.storage.sync_state import SyncStateRepository
from src.types import IndexAlias, IngestCursor, IngestStats, Language
from src.utils.logging import setup_logging
from src.utils.parsers import normalize_language, parse_duration, parse_pub_date

# All language-specific index aliases used by the v2 routing.
_ALIASES: tuple[str, ...] = ("episodes-zh-tw", "episodes-zh-cn", "episodes-en")

# Sentinel for distinguishing "not provided" from explicit None (BM25-only).
# run_incremental() and upsert_by_show_id() default embedding_backend to _UNSET,
# which is replaced with LocalEmbeddingBackend() at call time.
# Callers that pass None explicitly get BM25-only mode (vectors preserved).
_UNSET: object = object()

# Legacy v1 alias used when ENABLE_LANGUAGE_SPLIT=False.
_LEGACY_ALIAS = "episodes"

# Map raw SQLite target_index values to Language literals.
_TARGET_INDEX_TO_LANGUAGE: dict[str, Language] = {
    "podcast-episodes-zh-tw": "zh-tw",
    "podcast-episodes-zh-cn": "zh-cn",
    "podcast-episodes-en":    "en",
}

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


def _language_from_target_index(target_index: str) -> Optional[Language]:
    """Map a raw target_index value from SQLite to a Language literal.

    Args:
        target_index: Raw value stored in SQLite, e.g. "podcast-episodes-zh-tw".

    Returns:
        Language ("zh-tw", "zh-cn", or "en"), or None if unrecognised.
    """
    return _TARGET_INDEX_TO_LANGUAGE.get(target_index)


class EmbedAndIngestPipeline:
    """
    Pipeline to embed episode texts and ingest into Elasticsearch.

    Data flow:
        embedding_input/*.json + cleaned/*.json → EmbeddingBackend → language-specific ES alias

    The pipeline:
    1. Reads pre-built embedding_input files (title-weighted + truncated text)
    2. Loads corresponding cleaned episode data for metadata
    3. Groups episodes by language and generates embeddings via EmbeddingBackend
    4. Merges embedding into ES document
    5. Routes each episode to the correct language alias via IndexRoutingStrategy
    6. Bulk indexes into ES

    When embedding_backend is None, the pipeline runs in BM25-only mode:
    episodes are ingested without an embedding field (suitable for field-only
    backfills where existing vectors should be preserved by ES update).
    """

    EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")
    CLEANED_EPISODES_DIR = Path("data/cleaned/episodes")
    CLEANED_SHOWS_DIR = Path("data/cleaned/shows")

    def __init__(
        self,
        *,
        environment: str,
        es_service: Optional[ElasticsearchService] = None,
        embedding_backend: Optional[EmbeddingBackend] = None,
        batch_size: int = 64,
        dry_run: bool = False,
        storage: Optional[StorageBase] = None,
        routing_strategy: Optional[IndexRoutingStrategy] = None,
        allowed_show_ids: Optional[set[str]] = None,
        from_cache: bool = False,
        cache_dir: Optional[Path] = None,
        strict_cache: bool = False,
        sync_repo: Optional[SyncStateRepository] = None,
        episode_status_repo: Optional[EpisodeStatusRepository] = None,
        es_chunk_size: int = 500,
    ) -> None:
        self._environment = environment
        self.es = es_service or ElasticsearchService()
        self.batch_size = batch_size  # default raised to 64 for better GPU/MPS utilisation
        self._es_chunk_size = es_chunk_size
        self.dry_run = dry_run
        self.storage = storage or create_storage()
        self.routing_strategy: IndexRoutingStrategy = (
            routing_strategy or LanguageSplitRoutingStrategy()
        )
        self.allowed_show_ids = allowed_show_ids  # None = process all shows
        self._embedding_backend = embedding_backend  # None = BM25-only mode
        self._from_cache = from_cache
        self._cache_dir = cache_dir or settings.EMBEDDING_CACHE_DIR
        self._strict_cache = strict_cache
        self._sync_repo = sync_repo
        # Phase 2a CB1 per-show DB metadata commit lands here. When None, the
        # commit step is skipped — pipelines running without DB access (e.g.
        # some tests) simply omit the metadata write.
        self._episode_status_repo = episode_status_repo

        # Caches
        self._show_cache: Dict[str, Dict] = {}
        self._cleaned_episode_cache: Dict[str, Dict] = {}
        # episode_id → vector, populated when from_cache=True
        self._vector_cache: Dict[str, list] = {}

        # Ingest tracking (populated during build_actions)
        self._index_counts: Dict[str, int] = defaultdict(int)
        self._language_distribution: Dict[str, int] = defaultdict(int)
        # episode_id → index alias, populated during to_es_doc (used for writeback)
        self._episode_aliases: Dict[str, str] = {}

        # Cache hit/miss counters (populated during batch_encode when from_cache=True)
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # Phase 2a per-show cache outcome counters — populated during
        # _load_vector_cache when from_cache=True. Each candidate show lands
        # in exactly one of {cache_hit, cache_miss, cache_identity_mismatch}
        # plus an overlapping fallback_rebuild count = miss + mismatch.
        self._cache_hit_count: int = 0
        self._cache_miss_count: int = 0
        self._cache_identity_mismatch_count: int = 0
        self._fallback_rebuild_count: int = 0
        # Per-show rebuild outcomes — kept for the upcoming per-show DB
        # metadata commit (populated but not yet consumed in this batch).
        self._rebuild_results: Dict[str, ShowRebuildResult] = {}
        self._rebuild_failures: list[Dict] = []

    def _load_show_cache(self) -> None:
        """Pre-load show data from SQLiteStorage into memory.

        When allowed_show_ids is set, only those shows are loaded.
        """
        for show in self.storage.get_shows():
            if self.allowed_show_ids is None or show.show_id in self.allowed_show_ids:
                self._show_cache[show.show_id] = {
                    "show_id":      show.show_id,
                    "title":        show.title,
                    "author":       show.author,
                    "image_url":    show.image_url,
                    "external_urls": dict(show.external_urls),
                    "target_index": show.target_index,
                }
        logger.debug("show_cache_loaded", extra={"count": len(self._show_cache)})

    def _load_cleaned_episode_cache(self) -> None:
        """Pre-load all cleaned episode data into memory using parallel I/O."""
        if not self.CLEANED_EPISODES_DIR.exists():
            logger.warning(
                "cleaned_episodes_dir_not_found",
                extra={"path": str(self.CLEANED_EPISODES_DIR)},
            )
            return

        paths = list(self.CLEANED_EPISODES_DIR.glob("*.json"))

        def _read_one(path: Path) -> Optional[Dict]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    "cleaned_episode_load_failed",
                    extra={"file": str(path), "error": str(e)},
                )
                return None

        loaded = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            for episode_data in pool.map(_read_one, paths):
                if episode_data is None:
                    continue
                episode_id = episode_data.get("episode_id")
                show_id = episode_data.get("show_id")
                if episode_id and (
                    self.allowed_show_ids is None
                    or show_id in self.allowed_show_ids
                ):
                    self._cleaned_episode_cache[episode_id] = episode_data
                    loaded += 1

        logger.debug("cleaned_episode_cache_loaded", extra={"count": loaded})

    def _load_vector_cache(self) -> None:
        """Load pre-computed embedding vectors from the versioned cache layout.

        Per-show flow (Phase 2a §4.1):
          1. Resolve the expected identity for the show's language.
          2. Look up the versioned cache path; if missing, count a cache_miss
             and fall back to `rebuild_show_cache` (when a backend is available).
          3. If the cache exists, validate its identity against the expected.
             On match: count a cache_hit and populate `self._vector_cache`.
             On mismatch: count it, log a drift event, and fall back to
             `rebuild_show_cache`.
          4. `rebuild_show_cache` raises `EmbeddingDimensionContractViolation`
             for dimension violations — that propagates out of this method
             and halts the pipeline (systemic halt).

        Counters land exclusively in one bucket per show:
          cache_hit_count / cache_miss_count / cache_identity_mismatch_count.
        `fallback_rebuild_count` overlaps (miss + mismatch for shows where
        rebuild was attempted).

        Shows whose cache could not be loaded or rebuilt are logged but do
        not stop the load — their episodes will simply have no vectors
        (BM25-only update path in `to_es_doc`), matching pre-Phase-2a
        behavior.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        rebuild_supported = self._embedding_backend is not None
        loaded_episodes = 0
        show_ids = (
            self.allowed_show_ids
            if self.allowed_show_ids is not None
            else {
                ep.get("show_id")
                for ep in self._cleaned_episode_cache.values()
                if ep.get("show_id")
            }
        )

        for show_id in show_ids:
            identity_and_lang = self._resolve_identity_for_show(show_id)
            if identity_and_lang is None:
                logger.warning(
                    "cache_identity_unresolvable",
                    extra={"show_id": show_id},
                )
                continue
            identity, language = identity_and_lang

            cache_path = cache_path_for(self._cache_dir, identity, show_id)

            entry: Optional[Dict] = None
            rebuild_reason: Optional[str] = None

            if not cache_path.exists():
                self._cache_miss_count += 1
                rebuild_reason = "cache_miss"
            else:
                try:
                    with open(cache_path, encoding="utf-8") as fh:
                        entry = json.load(fh)
                except Exception as exc:  # noqa: BLE001 — unreadable file = treat as miss
                    logger.warning(
                        "embedding_cache_load_failed",
                        extra={"show_id": show_id, "error": repr(exc)},
                    )
                    self._cache_miss_count += 1
                    rebuild_reason = "cache_miss"
                    entry = None

                if entry is not None:
                    mismatch = validate_cache_identity(entry, identity)
                    if mismatch is None:
                        # Cache hit — three prohibitions: no cache rewrite, no
                        # DB metadata update, no runtime call. Just read.
                        self._cache_hit_count += 1
                        for episode_id, vector in (entry.get("episodes") or {}).items():
                            self._vector_cache[episode_id] = vector
                            loaded_episodes += 1
                    else:
                        self._cache_identity_mismatch_count += 1
                        logger.warning(
                            "cache_identity_mismatch_detected",
                            extra={
                                "show_id": show_id,
                                "drift_kind": mismatch.drift_kind.value,
                                "found_parse_state": mismatch.found_parse_state.value,
                                "expected_model": mismatch.expected_model,
                                "expected_version": mismatch.expected_version,
                                "expected_dims": mismatch.expected_dims,
                                "found_model": mismatch.found_model,
                                "found_version": mismatch.found_version,
                                "found_dims": mismatch.found_dims,
                                "vector_length_observed": mismatch.vector_length_observed,
                            },
                        )
                        rebuild_reason = "identity_mismatch"

            if rebuild_reason is None:
                continue  # cache hit path — nothing more to do for this show

            if not rebuild_supported:
                # BM25-only + from_cache=True with no usable cache — skip this
                # show's vectors; caller opted out of having a backend.
                logger.warning(
                    "cache_fallback_rebuild_skipped_no_backend",
                    extra={"show_id": show_id, "reason": rebuild_reason},
                )
                continue

            self._fallback_rebuild_count += 1
            rebuild_result = rebuild_show_cache(
                show_id=show_id,
                identity=identity,
                language=language,
                cache_dir=self._cache_dir,
                embedding_input_dir=self.EMBEDDING_INPUT_DIR,
                backend=self._embedding_backend,
            )

            if rebuild_result.status == "failed":
                self._rebuild_failures.append({
                    "show_id": show_id,
                    "error_code": rebuild_result.error_code,
                    "error_message": rebuild_result.error_message,
                    "rebuild_reason": rebuild_reason,
                })
                logger.warning(
                    "cache_fallback_rebuild_failed",
                    extra={
                        "show_id": show_id,
                        "error_code": rebuild_result.error_code,
                        "reason": rebuild_reason,
                    },
                )
                continue

            # Track the successful rebuild for the upcoming per-show DB
            # metadata commit (consumed in a later batch).
            self._rebuild_results[show_id] = rebuild_result

            # Re-read the freshly-written versioned cache to populate the
            # in-memory vector cache. Re-reading is the simplest way to keep
            # this method the single source of truth for vector population.
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    fresh = json.load(fh)
                for episode_id, vector in (fresh.get("episodes") or {}).items():
                    self._vector_cache[episode_id] = vector
                    loaded_episodes += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "cache_fallback_post_read_failed",
                    extra={"show_id": show_id, "error": repr(exc)},
                )

        logger.info(
            "vector_cache_loaded",
            extra={
                "episodes": loaded_episodes,
                "cache_hit_count": self._cache_hit_count,
                "cache_miss_count": self._cache_miss_count,
                "cache_identity_mismatch_count": self._cache_identity_mismatch_count,
                "fallback_rebuild_count": self._fallback_rebuild_count,
                "rebuild_failures": len(self._rebuild_failures),
            },
        )

    def _resolve_identity_for_show(
        self, show_id: str,
    ) -> Optional[tuple[EmbeddingIdentity, Language]]:
        """Return (identity, language) for a show, or None when unresolvable.

        Language is read from the show's target_index (the primary routing
        truth). Falls back to the cleaned episode cache when the show is not
        present in the show cache (e.g. allowed_show_ids contains IDs not in
        the current window).
        """
        show = self._show_cache.get(show_id)
        target_index = show.get("target_index") if show else None
        language = _language_from_target_index(target_index or "")
        if language is None:
            for ep in self._cleaned_episode_cache.values():
                if ep.get("show_id") == show_id:
                    language = _language_from_target_index(ep.get("target_index", ""))
                    if language is not None:
                        break
        if language is None:
            return None
        return resolve_expected_identity(language=language), language

    def _get_show_data(self, show_id: str) -> Optional[Dict]:
        """Get show data from cache."""
        return self._show_cache.get(show_id)

    def _get_cleaned_episode(self, episode_id: str) -> Optional[Dict]:
        """Get cleaned episode data from cache."""
        return self._cleaned_episode_cache.get(episode_id)

    def _get_language_for_input(self, inp: Dict) -> Optional[Language]:
        """Derive the Language for an embedding input from its cleaned episode cache entry.

        Looks up target_index in the cleaned episode cache and maps it to a Language.

        Args:
            inp: An embedding input dict with an 'episode_id' key.

        Returns:
            Language literal, or None if episode is unknown or target_index is unmapped.
        """
        episode_id = inp.get("episode_id")
        if not episode_id:
            return None
        cleaned = self._cleaned_episode_cache.get(episode_id)
        if not cleaned:
            return None
        return _language_from_target_index(cleaned.get("target_index", ""))

    def list_embedding_input_files(self) -> list[Path]:
        """List all embedding input JSON files."""
        if not self.EMBEDDING_INPUT_DIR.exists():
            logger.warning(
                "embedding_input_dir_not_found",
                extra={"path": str(self.EMBEDDING_INPUT_DIR)},
            )
            return []
        return sorted(self.EMBEDDING_INPUT_DIR.glob("*.json"))

    def load_embedding_inputs(self) -> Generator[Dict, None, None]:
        """Load embedding input files in parallel, filtered to allowed_show_ids when set."""
        files = self.list_embedding_input_files()

        def _read_one(f: Path) -> Optional[Dict]:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                logger.warning(
                    "embedding_input_load_failed",
                    extra={"file": str(f), "error": str(e)},
                )
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            for data in pool.map(_read_one, files):
                if data and (
                    self.allowed_show_ids is None
                    or data.get("show_id") in self.allowed_show_ids
                ):
                    yield data

    def batch_encode(
        self, inputs: list[Dict]
    ) -> list[tuple[Dict, list[float]]]:
        """Encode a batch of embedding inputs, grouping by language for efficiency.

        Groups inputs by their language (derived from target_index in the cleaned
        episode cache) and calls embedding_backend.embed_batch() once per language
        group. This allows LocalEmbeddingBackend to use a single model.encode(texts)
        call per language rather than N individual encode calls.

        Args:
            inputs: List of embedding input dicts, each with 'episode_id' and
                    'embedding_input.text' keys.

        Returns:
            List of (input_dict, embedding_vector) tuples in the same order as
            the inputs. Vectors are [] for items with unknown language or when
            embedding_backend is None (BM25-only mode).
        """
        if self._from_cache:
            result = []
            miss_count = 0
            for inp in inputs:
                episode_id = inp.get("episode_id", "")
                vec = self._vector_cache.get(episode_id, [])
                if vec:
                    self._cache_hits += 1
                else:
                    miss_count += 1
                    self._cache_misses += 1
                    logger.debug(
                        "vector_cache_miss",
                        extra={"episode_id": episode_id},
                    )
                result.append((inp, vec))
            if miss_count > 0 and self._strict_cache:
                raise CacheMissError(
                    f"{miss_count} cache miss(es) in strict-cache mode"
                )
            return result

        if self._embedding_backend is None:
            return [(inp, []) for inp in inputs]

        # Determine language and group input indices by language
        lang_groups: dict[Language, list[int]] = defaultdict(list)
        unknown_indices: list[int] = []

        for i, inp in enumerate(inputs):
            lang = self._get_language_for_input(inp)
            if lang is None:
                unknown_indices.append(i)
            else:
                lang_groups[lang].append(i)

        # Result buffer — pre-filled with empty vectors, filled in per group
        result: list[tuple[Dict, list[float]]] = [(inp, []) for inp in inputs]

        for lang, indices in lang_groups.items():
            texts = [inputs[i]["embedding_input"]["text"] for i in indices]
            vectors = self._embedding_backend.embed_batch(texts, lang)
            for i, vec in zip(indices, vectors):
                result[i] = (inputs[i], vec)

        return result

    def to_es_doc(
        self,
        embedding_input: Dict,
        embedding_vector: list[float],
    ) -> Optional[Dict]:
        """
        Build ES document from embedding_input, cleaned episode data, and embedding vector.

        Uses cleaned episode data for metadata instead of canonical storage.
        When embedding_vector is empty, the 'embedding' field is omitted from _source
        (BM25-only mode — preserves any existing vector in ES via upsert).
        """
        episode_id = embedding_input["episode_id"]
        show_id = embedding_input.get("show_id")

        # Load cleaned episode data
        cleaned_ep = self._get_cleaned_episode(episode_id)
        if not cleaned_ep:
            logger.warning(
                "cleaned_episode_not_found",
                extra={"episode_id": episode_id},
            )
            return None

        # Extract data from cleaned episode
        normalized = cleaned_ep.get("cleaned", {}).get("normalized", {})
        original_meta = cleaned_ep.get("original_meta", {})

        title = normalized.get("title")
        description = normalized.get("description")

        # Parse metadata from original_meta
        published_at = parse_pub_date(original_meta.get("pub_date"))
        duration_sec = parse_duration(original_meta.get("duration"))
        audio_url = original_meta.get("audio_url")
        language = normalize_language(original_meta.get("language"))

        # Get show data
        show_obj = {"show_id": show_id}
        if show_id:
            show_data = self._get_show_data(show_id)
            if show_data:
                # Support both raw storage format and cleaned format
                if "cleaned" in show_data:
                    show_normalized = show_data.get("cleaned", {}).get("normalized", {})
                    show_obj["title"] = show_normalized.get("title")
                    show_obj["publisher"] = show_normalized.get("author")
                else:
                    # Raw storage format (title/author at top level)
                    show_obj["title"] = show_data.get("title")
                    show_obj["publisher"] = show_data.get("author")
                    show_obj["image_url"] = show_data.get("image_url")
                    show_obj["external_urls"] = show_data.get("external_urls") or {}

        if settings.ENABLE_LANGUAGE_SPLIT:
            raw_target = cleaned_ep.get("target_index") or ""
            # Fallback: if cleaned JSON has no target_index (written before v2 language
            # detection), look it up from the show cache (current SQLite value).
            if not raw_target and show_id:
                show_data = self._get_show_data(show_id)
                raw_target = (show_data or {}).get("target_index") or ""
            if not raw_target:
                # Show has target_index=NULL in SQLite (unsupported language such as
                # ja/ko, or not yet classified). SQLiteStorage._WHERE_BASE excludes
                # these shows from the cache, so this is expected — not an error.
                logger.debug(
                    "episode_skipped_no_target_index",
                    extra={"episode_id": episode_id},
                )
                return None
            try:
                index_alias: str = self.routing_strategy.get_alias(raw_target)
            except ValueError:
                # target_index has a value but it is not in the routing map —
                # this is a genuine unexpected state worth investigating.
                logger.warning(
                    "episode_routing_failed",
                    extra={"episode_id": episode_id, "target_index": raw_target},
                )
                return None
        else:
            # v1 mode: route all episodes to the legacy monolithic alias.
            index_alias = _LEGACY_ALIAS

        # Track per-alias and per-language counts, and record alias for writeback
        self._index_counts[index_alias] += 1
        self._language_distribution[language or "unknown"] += 1
        self._episode_aliases[episode_id] = index_alias

        now = datetime.now(timezone.utc).isoformat()

        source: Dict = {
            "episode_id": episode_id,

            # Content (from cleaned data)
            "title": title,
            "description": description,

            # Metadata
            "published_at": published_at,
            "duration_sec": duration_sec,
            "language": language,

            # Audio (minimal info from original_meta)
            "audio": {
                "url": audio_url,
            },

            # Show
            "show": show_obj,

            # Timestamps
            "created_at": now,
            "updated_at": now,
        }

        if embedding_vector:
            source["embedding"] = embedding_vector
            return {
                "_index": index_alias,
                "_id": episode_id,
                "_source": source,
            }

        # BM25-only mode: preserve any existing embedding vector and created_at timestamp.
        # doc omits created_at so it is not overwritten on existing documents;
        # upsert (first-insert path) includes created_at.
        update_doc = {k: v for k, v in source.items() if k != "created_at"}
        return {
            "_op_type": "update",
            "_index": index_alias,
            "_id": episode_id,
            "doc": update_doc,
            "upsert": source,
        }

    def build_actions(
        self, inputs: Iterable[Dict]
    ) -> Generator[Dict, None, None]:
        """
        Build ES bulk actions from embedding inputs.

        Batches encoding for efficiency.
        """
        batch = []

        for inp in inputs:
            batch.append(inp)

            if len(batch) >= self.batch_size:
                encoded = self.batch_encode(batch)
                for inp_data, emb in encoded:
                    doc = self.to_es_doc(inp_data, emb)
                    if doc:
                        yield doc
                batch = []

        # Process remaining
        if batch:
            encoded = self.batch_encode(batch)
            for inp_data, emb in encoded:
                doc = self.to_es_doc(inp_data, emb)
                if doc:
                    yield doc

    def run(self) -> Dict:
        """
        Run the embed and ingest pipeline.

        Returns stats dict.
        """
        start_time = datetime.now(timezone.utc)

        # Load caches
        self._load_show_cache()
        self._load_cleaned_episode_cache()
        if self._from_cache:
            self._load_vector_cache()

        # Count files
        files = self.list_embedding_input_files()
        total_count = len(files)

        logger.info(
            "pipeline_start",
            extra={
                "total_files": total_count,
                "batch_size": self.batch_size,
                "dry_run": self.dry_run,
                "embedding": self._embedding_backend is not None,
            },
        )

        if not files:
            logger.warning("no_embedding_input_files")
            return {"success": 0, "errors": 0, "total": 0}

        if self.dry_run:
            logger.info("dry_run_mode")
            sample_inputs = list(self.load_embedding_inputs())[:3]
            for inp in sample_inputs:
                logger.info(
                    "sample_input",
                    extra={
                        "episode_id": inp["episode_id"],
                        "text_len": len(inp["embedding_input"]["text"]),
                    },
                )
            return {"success": 0, "errors": 0, "total": total_count, "dry_run": True}

        # Run bulk ingest
        success = 0
        errors: list = []
        successful_ids: list[str] = []

        # Phase 2a §3.7: per-show bulk outcome tally. `show_bulk_ok` for a
        # show is true iff every one of its actions came back ok AND at
        # least one action was attempted. Used downstream for CB1 per-show
        # DB metadata commit and OB1/OB2 processed/failed partitioning.
        show_bulk_tally: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"ok": 0, "err": 0}
        )
        show_successful_ids: Dict[str, list[str]] = defaultdict(list)

        # §4.0 Error funneling: wrap bulk + transform to ensure A1 coverage.
        # streaming_bulk(client, self.build_actions(...), ...) — argument evaluation
        # happens inside the try, so transform-time errors (build_actions /
        # load_embedding_inputs raising) are also caught here.
        try:
            with tqdm(desc="Syncing to ES", unit="ep") as pbar:
                for ok, item in streaming_bulk(
                    self.es.client,
                    self.build_actions(self.load_embedding_inputs()),
                    chunk_size=self._es_chunk_size,
                    raise_on_error=False,
                ):
                    op_type = next(iter(item))
                    ep_id = item[op_type].get("_id")
                    show_id = None
                    if ep_id:
                        ep_record = self._cleaned_episode_cache.get(ep_id)
                        if ep_record:
                            show_id = ep_record.get("show_id")
                    if ok:
                        success += 1
                        if ep_id:
                            successful_ids.append(ep_id)
                        if show_id:
                            show_bulk_tally[show_id]["ok"] += 1
                            if ep_id:
                                show_successful_ids[show_id].append(ep_id)
                    else:
                        errors.append(item)
                        if show_id:
                            show_bulk_tally[show_id]["err"] += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=success, err=len(errors))
        except Exception as exc:
            errors.append({"type": "bulk_or_transform_exception", "error": repr(exc)})
            logger.exception("bulk_or_transform_exception")

        # §4.2 R3 all-or-nothing flush: any step-level error → no sync_state writeback.
        if errors:
            logger.warning(
                "sync_state_flush_skipped_due_to_errors",
                extra={
                    "error_count": len(errors),
                    "would_have_flushed": len(successful_ids),
                },
            )
        elif successful_ids and self._sync_repo is not None:
            try:
                for ep_id in successful_ids:
                    alias = self._episode_aliases.get(ep_id, "")
                    self._sync_repo.mark_done(
                        entity_type="episode",
                        entity_id=ep_id,
                        index_alias=alias,
                        environment=self._environment,
                    )
                self._sync_repo.commit()
                logger.info(
                    "writeback_complete",
                    extra={"count": len(successful_ids), "environment": self._environment},
                )
            except Exception as exc:
                # Release any partial writes held in SQLite's implicit transaction.
                # Without this, a long-lived connection would carry stale pending
                # rows into the next caller's commit() and silently break R3.
                try:
                    self._sync_repo._db.conn.rollback()
                except Exception:
                    logger.exception("flush_rollback_failed")
                errors.append({"type": "flush_exception", "error": repr(exc)})
                logger.exception("flush_exception")

        # Phase 2a §3.7 CB1: per-show DB metadata commit. Commit is triggered
        # independently per show — a partial-failure run still commits the
        # metadata of shows whose rebuild + bulk both succeeded, so that the
        # next run's drift detector does not re-trigger rebuild for those
        # shows. This is deliberately NOT gated by the run-level `errors`
        # check (that gate belongs to sync_state; see Phase 1 R3).
        committed_shows: list[str] = []
        if self._episode_status_repo is not None:
            for show_id, rebuild_result in self._rebuild_results.items():
                if rebuild_result.status != "ok":
                    continue
                tally = show_bulk_tally.get(show_id, {"ok": 0, "err": 0})
                show_bulk_ok = tally["err"] == 0 and tally["ok"] > 0
                if not show_bulk_ok:
                    continue
                show_episode_ids = show_successful_ids.get(show_id, [])
                if not show_episode_ids:
                    continue
                try:
                    self._episode_status_repo.mark_embedding_metadata_only(
                        episode_ids=show_episode_ids,
                        model=rebuild_result.identity_used.model_name,
                        version=rebuild_result.identity_used.embedding_version,
                        embedded_at=rebuild_result.new_last_embedded_at.isoformat(),
                    )
                    committed_shows.append(show_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "cb1_metadata_commit_failed",
                        extra={"show_id": show_id, "error": repr(exc)},
                    )

        # Phase 2a §3.7 OB1/OB2: partition candidate shows into processed /
        # failed. A show is `processed` iff (cache hit OR rebuild ok) AND
        # show_bulk_ok (every one of its bulk actions succeeded). Any other
        # state counts as `failed` for this phase's strict OB1 definition.
        candidate_shows: set[str] = (
            set(self.allowed_show_ids)
            if self.allowed_show_ids is not None
            else set(show_bulk_tally.keys()) | set(self._rebuild_results.keys())
        )
        processed_count = 0
        failed_count = 0
        for sid in candidate_shows:
            tally = show_bulk_tally.get(sid, {"ok": 0, "err": 0})
            show_bulk_ok = tally["err"] == 0 and tally["ok"] > 0
            rebuild_result = self._rebuild_results.get(sid)
            had_cache_hit = (
                rebuild_result is None and sid not in {
                    f["show_id"] for f in self._rebuild_failures
                }
                and tally["ok"] + tally["err"] > 0
            )
            rebuild_ok = rebuild_result is not None and rebuild_result.status == "ok"
            if show_bulk_ok and (rebuild_ok or had_cache_hit):
                processed_count += 1
            else:
                failed_count += 1

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        stats = {
            "success": success,
            "errors": len(errors),
            "total": total_count,
            "elapsed_sec": round(elapsed, 2),
            "docs_per_sec": round(success / elapsed, 2) if elapsed > 0 else 0,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            # Phase 2a per-show cache outcome counters (§3.7 RA1–RA4).
            "cache_hit_count": self._cache_hit_count,
            "cache_miss_count": self._cache_miss_count,
            "cache_identity_mismatch_count": self._cache_identity_mismatch_count,
            "fallback_rebuild_count": self._fallback_rebuild_count,
            "rebuild_failures": len(self._rebuild_failures),
            # Phase 2a §3.7 OB1/OB2: per-show outcome partition. Invariant:
            # processed + failed == len(candidate shows seen in this run).
            "processed_shows": processed_count,
            "failed_shows": failed_count,
            # Phase 2a §3.7 CB1: shows that received a per-show metadata
            # commit (rebuild ok + bulk ok). Cache-hit shows are NOT in here.
            "committed_shows": len(committed_shows),
        }
        if self._cache_misses > 0:
            logger.warning(
                "cache_miss_warning",
                extra={"cache_misses": self._cache_misses},
            )

        logger.info("pipeline_complete", extra=stats)

        if errors:
            logger.warning(
                "ingest_errors_sample",
                extra={"sample": errors[:5]},
            )

        emit_ingest_log(
            index_counts=dict(self._index_counts),
            language_distribution=dict(self._language_distribution),
            ingest_success=success,
            ingest_failed=len(errors),
        )

        return stats


def load_cursor(path: Optional[Path] = None) -> dict[str, IngestCursor]:
    """Load per-index ingest cursor from a JSON file.

    Returns an empty dict when the file does not exist (first run = full ingest).

    Args:
        path: Path to the cursor file. Defaults to settings.INGEST_CURSOR_PATH.

    Returns:
        Mapping of alias name → IngestCursor, or {} if the file is missing.
    """
    _path = path or settings.INGEST_CURSOR_PATH
    if not _path.exists():
        return {}
    return json.loads(_path.read_text())


def save_cursor(
    cursors: dict[str, IngestCursor],
    path: Optional[Path] = None,
) -> None:
    """Persist per-index ingest cursor to a JSON file.

    Creates missing parent directories automatically.

    Args:
        cursors: Mapping of alias name → IngestCursor to persist.
        path:    Destination path. Defaults to settings.INGEST_CURSOR_PATH.
    """
    _path = path or settings.INGEST_CURSOR_PATH
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_text(json.dumps(cursors, indent=2))


def run_incremental(
    storage: Optional[StorageBase] = None,
    routing: Optional[IndexRoutingStrategy] = None,
    embedding_backend: Optional[EmbeddingBackend] = _UNSET,  # type: ignore[assignment]
    force_full: bool = False,
    cursor_path: Optional[Path] = None,
    batch_size: int = 64,
    dry_run: bool = False,
    allowed_show_ids: Optional[set[str]] = None,
    from_cache: bool = False,
    cache_dir: Optional[Path] = None,
    strict_cache: bool = False,
    sync_repo: Optional[SyncStateRepository] = None,
    episode_status_repo: Optional[EpisodeStatusRepository] = None,
    es_chunk_size: int = 500,
) -> dict:
    """Run ingest in incremental mode: only shows updated since the last run.

    Uses a per-alias cursor to determine which shows to process. On the first
    run (no cursor file), all shows are ingested. Saves an updated cursor after
    a successful run.

    Args:
        storage:           Storage backend. Defaults to create_storage().
        routing:           Index routing strategy. Defaults to LanguageSplitRoutingStrategy.
        embedding_backend: Embedding backend.
                           Omit (default) → LocalEmbeddingBackend() is used.
                           Pass None explicitly → BM25-only mode; existing ES vectors
                           are preserved via update semantics.
        force_full:        Ignore the cursor and process all shows (e.g. after a
                           mapping change). Equivalent to run_backfill().
        cursor_path:       Path to the cursor file. Defaults to settings.INGEST_CURSOR_PATH.
        batch_size:        Embedding batch size passed to EmbedAndIngestPipeline.
        dry_run:           Dry-run flag passed to EmbedAndIngestPipeline.
        allowed_show_ids:  Optional set of show IDs to restrict processing to.
                           When provided with force_full=True, only those shows are
                           processed. When provided with force_full=False, only the
                           intersection with cursor-updated shows is processed.

    Returns:
        Stats dict from EmbedAndIngestPipeline.run(), or
        {"success": 0, "errors": 0, "total": 0} when no shows are updated.
    """
    _storage = storage or create_storage()
    _backend: Optional[EmbeddingBackend] = (
        LocalEmbeddingBackend() if embedding_backend is _UNSET else embedding_backend
    )
    cursors = {} if force_full else load_cursor(cursor_path)
    now = datetime.now(timezone.utc).isoformat()

    if force_full:
        _allowed_show_ids: Optional[set[str]] = allowed_show_ids
    else:
        since_values = [c.get("last_ingest_at", "") for c in cursors.values()]
        since = min(since_values) if since_values else ""
        updated = list(_storage.get_shows_updated_since(since))
        if not updated:
            # Phase 1 R7: no-op run leaves zero footprint on the cursor —
            # both last_ingest_at and last_run_at are frozen.
            logger.info("incremental_no_updates_noop", extra={"since": since})
            return {"success": 0, "errors": 0, "total": 0}
        updated_ids = {show.show_id for show in updated}
        _allowed_show_ids = (
            updated_ids & allowed_show_ids if allowed_show_ids is not None else updated_ids
        )
        logger.info("incremental_updated_shows", extra={"count": len(_allowed_show_ids), "since": since})

    pipeline = EmbedAndIngestPipeline(
        environment="local",  # §3.1.1 rule F: daily ingest path writes sync_state under 'local'
        storage=_storage,
        routing_strategy=routing or LanguageSplitRoutingStrategy(),
        embedding_backend=_backend,
        allowed_show_ids=_allowed_show_ids,
        batch_size=batch_size,
        dry_run=dry_run,
        from_cache=from_cache,
        cache_dir=cache_dir,
        strict_cache=strict_cache,
        sync_repo=sync_repo,
        episode_status_repo=episode_status_repo,
        es_chunk_size=es_chunk_size,
    )
    stats = pipeline.run()

    # Phase 1 R1: cursor advances only when the whole step succeeded.
    # Sev0 fix: partial failure must not advance cursor.
    if stats.get("errors", 0) == 0:
        _cursor_aliases = _ALIASES if settings.ENABLE_LANGUAGE_SPLIT else (_LEGACY_ALIAS,)
        save_cursor(
            {alias: IngestCursor(last_ingest_at=now, last_run_at=now) for alias in _cursor_aliases},
            cursor_path,
        )
    else:
        logger.warning(
            "cursor_not_advanced_due_to_errors",
            extra={"error_count": stats["errors"], "success_count": stats.get("success", 0)},
        )
    return stats


def run_backfill(
    storage: Optional[StorageBase] = None,
    routing: Optional[IndexRoutingStrategy] = None,
    embedding_backend: Optional[EmbeddingBackend] = None,
    cursor_path: Optional[Path] = None,
    batch_size: int = 64,
    dry_run: bool = False,
    sync_repo: Optional[SyncStateRepository] = None,
    episode_status_repo: Optional[EpisodeStatusRepository] = None,
    es_chunk_size: int = 500,
) -> dict:
    """Re-ingest all shows regardless of cursor (force_full=True).

    Use this after adding a non-analyzer field to the mapping: it re-ingests
    every document without re-computing embeddings (embedding_backend defaults
    to None so existing ES vectors are preserved via update semantics).

    To also re-embed during backfill, pass an explicit embedding_backend.

    Args:
        storage:           Storage backend. Defaults to create_storage().
        routing:           Index routing strategy.
        embedding_backend: Embedding backend. Defaults to None (preserve vectors).
        cursor_path:       Path to the cursor file.
        batch_size:        Embedding batch size.
        dry_run:           Dry-run flag.

    Returns:
        Stats dict from EmbedAndIngestPipeline.run().
    """
    return run_incremental(
        storage=storage,
        routing=routing,
        embedding_backend=embedding_backend,
        force_full=True,
        cursor_path=cursor_path,
        batch_size=batch_size,
        dry_run=dry_run,
        sync_repo=sync_repo,
        episode_status_repo=episode_status_repo,
        es_chunk_size=es_chunk_size,
    )


def upsert_by_show_id(
    show_id: str,
    storage: Optional[StorageBase] = None,
    routing: Optional[IndexRoutingStrategy] = None,
    embedding_backend: Optional[EmbeddingBackend] = _UNSET,  # type: ignore[assignment]
    batch_size: int = 64,
    dry_run: bool = False,
    sync_repo: Optional[SyncStateRepository] = None,
    episode_status_repo: Optional[EpisodeStatusRepository] = None,
    es_chunk_size: int = 500,
) -> int:
    """Re-ingest all episodes for a single show.

    Useful for targeted fixes when a single show's data is incorrect or
    missing, without running the full backfill.

    Args:
        show_id:           The show to re-ingest.
        storage:           Storage backend. Defaults to create_storage().
        routing:           Index routing strategy.
        embedding_backend: Embedding backend.
                           Omit (default) → LocalEmbeddingBackend() is used.
                           Pass None explicitly → BM25-only mode; existing vectors
                           are preserved via update semantics.
        batch_size:        Embedding batch size.
        dry_run:           Dry-run flag.

    Returns:
        Number of episodes successfully ingested.

    Raises:
        ValueError: If show_id is not found in storage.
    """
    _storage = storage or create_storage()
    _show = next((s for s in _storage.get_shows() if s.show_id == show_id), None)
    if _show is None:
        raise ValueError(f"show_id not found in storage: {show_id!r}")

    _backend: Optional[EmbeddingBackend] = (
        LocalEmbeddingBackend() if embedding_backend is _UNSET else embedding_backend
    )

    pipeline = EmbedAndIngestPipeline(
        environment=settings.ES_ENV,  # single/upsert path keeps legacy behavior; Phase 1 scope excludes this
        storage=_storage,
        routing_strategy=routing or LanguageSplitRoutingStrategy(),
        embedding_backend=_backend,
        allowed_show_ids={show_id},
        batch_size=batch_size,
        dry_run=dry_run,
        sync_repo=sync_repo,
        episode_status_repo=episode_status_repo,
        es_chunk_size=es_chunk_size,
    )
    stats = pipeline.run()
    return stats.get("success", 0)


def emit_ingest_log(
    index_counts: Dict[str, int],
    language_distribution: Dict[str, int],
    ingest_success: int,
    ingest_failed: int,
) -> None:
    """Emit a structured ingest completion log entry.

    Logs a WARNING if uncertain_rate exceeds 5%, which may indicate
    a language detection regression in the crawler.

    Args:
        index_counts:          Document count per ES alias.
        language_distribution: Count per detected language (incl. "unknown").
        ingest_success:        Number of documents successfully indexed.
        ingest_failed:         Number of documents that failed to index.
    """
    total = ingest_success + ingest_failed
    uncertain_rate = (
        language_distribution.get("unknown", 0) / total if total > 0 else 0.0
    )
    stats: IngestStats = {
        "event":                 "ingest_complete",
        "timestamp":             datetime.now(timezone.utc).isoformat(),
        "index_counts":          index_counts,
        "language_distribution": language_distribution,
        "ingest_success":        ingest_success,
        "ingest_failed":         ingest_failed,
        "uncertain_rate":        uncertain_rate,
    }
    logger.info(json.dumps(stats))
    if uncertain_rate > 0.05:
        logger.warning(
            "uncertain_rate_high",
            extra={"uncertain_rate": round(uncertain_rate, 4)},
        )


def run() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest episodes to ES")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-full", action="store_true", help="Ignore cursor, process all shows")
    parser.add_argument("--show-id", type=str, help="Show ID for single-show upsert (SYNC_MODE=single)")
    parser.add_argument("--show-ids", nargs="+", metavar="SHOW_ID", help="One or more show IDs to process (incremental/full modes)")
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help="Read embedding vectors from data/embeddings/ instead of computing them. "
             "Run embed_episodes first to populate the cache.",
    )
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="Fail if any episode is missing from the vector cache (requires --from-cache). "
             "Use in CI or backfill to ensure all episodes have been pre-embedded.",
    )
    parser.add_argument(
        "--es-chunk-size",
        type=int,
        default=500,
        help="Number of documents per ES bulk request (default: 500). "
             "Reduce to 100–200 for remote ES to avoid request timeouts.",
    )
    args = parser.parse_args()

    if args.strict_cache and not args.from_cache:
        raise SystemExit("--strict-cache requires --from-cache")

    setup_logging()

    mode = settings.SYNC_MODE
    # --from-cache: skip embedding entirely; vectors come from disk cache.
    backend: Optional[EmbeddingBackend] = None if args.from_cache else LocalEmbeddingBackend()

    # Share one Database handle across both repos so commits land on the
    # same connection and transaction semantics match Phase 1's writeback.
    _db = Database(settings.SQLITE_PATH)
    _sync_repo = SyncStateRepository(_db)
    # Phase 2a CB1: per-show DB metadata commit repo. Wired at the CLI
    # entry point so every production mode (single / backfill / incremental)
    # picks it up. Without this, the pipeline's per-show commit path is
    # silently skipped because the repo stays None.
    _episode_status_repo = EpisodeStatusRepository(_db)

    if mode == "single":
        if not args.show_id:
            raise SystemExit("--show-id is required when SYNC_MODE=single")
        count = upsert_by_show_id(
            args.show_id,
            embedding_backend=backend,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            sync_repo=_sync_repo,
            episode_status_repo=_episode_status_repo,
            es_chunk_size=args.es_chunk_size,
        )
        logger.info("upsert_complete", extra={"show_id": args.show_id, "episodes": count})
    elif mode == "backfill":
        # Backfill preserves existing vectors by default (embedding_backend=None)
        run_backfill(
            embedding_backend=None,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            sync_repo=_sync_repo,
            episode_status_repo=_episode_status_repo,
            es_chunk_size=args.es_chunk_size,
        )
    else:
        # incremental (default) or full — Phase 1 daily path.
        # Sev0 fix: propagate non-zero exit when any ingest error occurred.
        # single / backfill branches retain pre-Phase-1 exit semantics.
        show_ids_filter = set(args.show_ids) if args.show_ids else None
        stats = run_incremental(
            embedding_backend=backend,
            force_full=(mode == "full" or args.force_full),
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            allowed_show_ids=show_ids_filter,
            from_cache=args.from_cache,
            strict_cache=args.strict_cache,
            sync_repo=_sync_repo,
            episode_status_repo=_episode_status_repo,
            es_chunk_size=args.es_chunk_size,
        )
        if stats and stats.get("errors", 0) > 0:
            logger.error(
                "exit_nonzero_due_to_ingest_errors",
                extra={"errors": stats["errors"], "success": stats.get("success", 0)},
            )
            sys.exit(1)


if __name__ == "__main__":
    run()
