"""
Embed and Ingest Episodes Pipeline

Read embedding_input files, generate embeddings, and write to Elasticsearch.

Data sources:
- embedding_input: data/embedding_input/episodes/*.json (text for embedding)
- cleaned: data/cleaned/episodes/*.json (episode metadata)
- shows: data/cleaned/shows/*.json (show metadata)

Usage:
    python -m src.pipelines.embed_and_ingest
    python -m src.pipelines.embed_and_ingest --batch-size 64
    python -m src.pipelines.embed_and_ingest --dry-run
    python -m src.pipelines.embed_and_ingest --force-full      # ignore cursor
    python -m src.pipelines.embed_and_ingest --show-id <id>    # SYNC_MODE=single
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

from elasticsearch.helpers import streaming_bulk

from src.config import settings
from src.embedding.encoder import EmbeddingEncoder
from src.search.routing import IndexRoutingStrategy, LanguageSplitRoutingStrategy
from src.services.es_service import ElasticsearchService
from src.storage.base import StorageBase
from src.storage.factory import create_storage
from src.types import IndexAlias, IngestCursor, IngestStats
from src.utils.logging import setup_logging
from src.utils.parsers import normalize_language, parse_duration, parse_pub_date

# All language-specific index aliases used by the v2 routing.
_ALIASES: tuple[str, ...] = ("episodes-zh-tw", "episodes-zh-cn", "episodes-en")

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class EmbedAndIngestPipeline:
    """
    Pipeline to embed episode texts and ingest into Elasticsearch.

    Data flow:
        embedding_input/*.json + cleaned/*.json → EmbeddingEncoder → language-specific ES alias

    The pipeline:
    1. Reads pre-built embedding_input files (title-weighted + truncated text)
    2. Loads corresponding cleaned episode data for metadata
    3. Generates embeddings using sentence-transformers
    4. Merges embedding into ES document
    5. Routes each episode to the correct language alias via IndexRoutingStrategy
    6. Bulk indexes into ES
    """

    EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")
    CLEANED_EPISODES_DIR = Path("data/cleaned/episodes")
    CLEANED_SHOWS_DIR = Path("data/cleaned/shows")

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
        encoder: Optional[EmbeddingEncoder] = None,
        batch_size: int = 32,
        dry_run: bool = False,
        storage: Optional[StorageBase] = None,
        routing_strategy: Optional[IndexRoutingStrategy] = None,
        allowed_show_ids: Optional[set[str]] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        self.batch_size = batch_size
        self.dry_run = dry_run
        self.storage = storage or create_storage()
        self.routing_strategy: IndexRoutingStrategy = (
            routing_strategy or LanguageSplitRoutingStrategy()
        )
        self.allowed_show_ids = allowed_show_ids  # None = process all shows

        # Lazy load encoder (it's heavy)
        self._encoder = encoder

        # Caches
        self._show_cache: Dict[str, Dict] = {}
        self._cleaned_episode_cache: Dict[str, Dict] = {}

        # Ingest tracking (populated during build_actions)
        self._index_counts: Dict[str, int] = defaultdict(int)
        self._language_distribution: Dict[str, int] = defaultdict(int)

    @property
    def encoder(self) -> EmbeddingEncoder:
        """Lazy load the embedding encoder."""
        if self._encoder is None:
            logger.info("loading_encoder")
            self._encoder = EmbeddingEncoder()
            logger.info(
                "encoder_loaded",
                extra={
                    "model": self._encoder.model_name,
                    "dim": self._encoder.embedding_dim,
                    "device": str(self._encoder.model.device),
                },
            )
        return self._encoder

    def _load_show_cache(self) -> None:
        """Pre-load show data from SQLiteStorage into memory.

        When allowed_show_ids is set, only those shows are loaded.
        """
        for show in self.storage.get_shows():
            if self.allowed_show_ids is None or show.show_id in self.allowed_show_ids:
                self._show_cache[show.show_id] = {
                    "show_id": show.show_id,
                    "title": show.title,
                    "author": show.author,
                }
        logger.info("show_cache_loaded", extra={"count": len(self._show_cache)})

    def _load_cleaned_episode_cache(self) -> None:
        """Pre-load all cleaned episode data into memory."""
        if not self.CLEANED_EPISODES_DIR.exists():
            logger.warning(
                "cleaned_episodes_dir_not_found",
                extra={"path": str(self.CLEANED_EPISODES_DIR)},
            )
            return

        loaded = 0
        for path in self.CLEANED_EPISODES_DIR.glob("*.json"):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    episode_data = json.load(f)
                    episode_id = episode_data.get("episode_id")
                    show_id = episode_data.get("show_id")
                    if episode_id and (
                        self.allowed_show_ids is None
                        or show_id in self.allowed_show_ids
                    ):
                        self._cleaned_episode_cache[episode_id] = episode_data
                        loaded += 1
            except Exception as e:
                logger.warning(
                    "cleaned_episode_load_failed",
                    extra={"file": str(path), "error": str(e)},
                )

        logger.info(
            "cleaned_episode_cache_loaded",
            extra={"count": loaded},
        )

    def _get_show_data(self, show_id: str) -> Optional[Dict]:
        """Get show data from cache."""
        return self._show_cache.get(show_id)

    def _get_cleaned_episode(self, episode_id: str) -> Optional[Dict]:
        """Get cleaned episode data from cache."""
        return self._cleaned_episode_cache.get(episode_id)

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
        """Load embedding input files, filtered to allowed_show_ids when set."""
        files = self.list_embedding_input_files()
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                if self.allowed_show_ids is None or data.get("show_id") in self.allowed_show_ids:
                    yield data
            except Exception as e:
                logger.warning(
                    "embedding_input_load_failed",
                    extra={"file": str(f), "error": str(e)},
                )

    def batch_encode(
        self, inputs: list[Dict]
    ) -> list[tuple[Dict, list[float]]]:
        """
        Batch encode embedding inputs.

        Returns list of (input_dict, embedding_vector) tuples.
        """
        texts = [inp["embedding_input"]["text"] for inp in inputs]
        embeddings = self.encoder.encode_batch(texts, batch_size=self.batch_size)

        results = []
        for inp, emb in zip(inputs, embeddings):
            results.append((inp, self.encoder.to_list(emb)))
        return results

    def to_es_doc(
        self,
        embedding_input: Dict,
        embedding_vector: list[float],
    ) -> Optional[Dict]:
        """
        Build ES document from embedding_input, cleaned episode data, and embedding vector.

        Uses cleaned episode data for metadata instead of canonical storage.
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

                    # Add image_url from image.url
                    image = show_data.get("image") or {}
                    show_obj["image_url"] = image.get("url")

                    # Add external_urls
                    external_urls = show_data.get("external_urls") or {}
                    show_obj["external_urls"] = external_urls

        raw_target = cleaned_ep.get("target_index", "")
        try:
            index_alias: str = self.routing_strategy.get_alias(raw_target)
        except ValueError:
            logger.warning(
                "episode_routing_failed",
                extra={"episode_id": episode_id, "target_index": raw_target},
            )
            return None

        # Track per-alias and per-language counts
        self._index_counts[index_alias] += 1
        self._language_distribution[language or "unknown"] += 1

        now = datetime.now(timezone.utc).isoformat()

        return {
            "_index": index_alias,
            "_id": episode_id,
            "_source": {
                "episode_id": episode_id,

                # Content (from cleaned data)
                "title": title,
                "description": description,

                # Embedding
                "embedding": embedding_vector,

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
            },
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
                # Encode batch
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

        # Count files
        files = self.list_embedding_input_files()
        total_count = len(files)

        logger.info(
            "pipeline_start",
            extra={
                "total_files": total_count,
                "batch_size": self.batch_size,
                "dry_run": self.dry_run,
            },
        )

        if not files:
            logger.warning("no_embedding_input_files")
            return {"success": 0, "errors": 0, "total": 0}

        if self.dry_run:
            # Just count and sample
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
        errors = []

        for ok, item in streaming_bulk(
            self.es.client,
            self.build_actions(self.load_embedding_inputs()),
            chunk_size=500,
            raise_on_error=False,
        ):
            if ok:
                success += 1
                if success % 100 == 0:
                    logger.info(
                        "ingest_progress",
                        extra={"processed": success, "total": total_count},
                    )
            else:
                errors.append(item)

        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

        stats = {
            "success": success,
            "errors": len(errors),
            "total": total_count,
            "elapsed_sec": round(elapsed, 2),
            "docs_per_sec": round(success / elapsed, 2) if elapsed > 0 else 0,
        }

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
    force_full: bool = False,
    cursor_path: Optional[Path] = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> dict:
    """Run ingest in incremental mode: only shows updated since the last run.

    Uses a per-alias cursor to determine which shows to process. On the first
    run (no cursor file), all shows are ingested. Saves an updated cursor after
    a successful run.

    Args:
        storage:     Storage backend. Defaults to create_storage().
        routing:     Index routing strategy. Defaults to LanguageSplitRoutingStrategy.
        force_full:  Ignore the cursor and process all shows (e.g. after a
                     mapping change). Equivalent to run_backfill().
        cursor_path: Path to the cursor file. Defaults to settings.INGEST_CURSOR_PATH.
        batch_size:  Embedding batch size passed to EmbedAndIngestPipeline.
        dry_run:     Dry-run flag passed to EmbedAndIngestPipeline.

    Returns:
        Stats dict from EmbedAndIngestPipeline.run(), or
        {"success": 0, "errors": 0, "total": 0} when no shows are updated.
    """
    _storage = storage or create_storage()
    cursors = {} if force_full else load_cursor(cursor_path)
    now = datetime.now(timezone.utc).isoformat()

    if force_full:
        allowed_show_ids: Optional[set[str]] = None
    else:
        since_values = [c.get("last_ingest_at", "") for c in cursors.values()]
        since = min(since_values) if since_values else ""
        updated = list(_storage.get_shows_updated_since(since))
        if not updated:
            logger.info("incremental_no_updates", extra={"since": since})
            save_cursor(
                {alias: IngestCursor(last_ingest_at=now, last_run_at=now) for alias in _ALIASES},
                cursor_path,
            )
            return {"success": 0, "errors": 0, "total": 0}
        allowed_show_ids = {show.show_id for show in updated}
        logger.info("incremental_updated_shows", extra={"count": len(allowed_show_ids), "since": since})

    pipeline = EmbedAndIngestPipeline(
        storage=_storage,
        routing_strategy=routing or LanguageSplitRoutingStrategy(),
        allowed_show_ids=allowed_show_ids,
        batch_size=batch_size,
        dry_run=dry_run,
    )
    stats = pipeline.run()

    save_cursor(
        {alias: IngestCursor(last_ingest_at=now, last_run_at=now) for alias in _ALIASES},
        cursor_path,
    )
    return stats


def run_backfill(
    storage: Optional[StorageBase] = None,
    routing: Optional[IndexRoutingStrategy] = None,
    cursor_path: Optional[Path] = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> dict:
    """Re-ingest all shows regardless of cursor (force_full=True).

    Use this after adding a non-analyzer field to the mapping: it re-ingests
    every document without re-computing embeddings.

    Args:
        storage:     Storage backend. Defaults to create_storage().
        routing:     Index routing strategy.
        cursor_path: Path to the cursor file.
        batch_size:  Embedding batch size.
        dry_run:     Dry-run flag.

    Returns:
        Stats dict from EmbedAndIngestPipeline.run().
    """
    return run_incremental(
        storage=storage,
        routing=routing,
        force_full=True,
        cursor_path=cursor_path,
        batch_size=batch_size,
        dry_run=dry_run,
    )


def upsert_by_show_id(
    show_id: str,
    storage: Optional[StorageBase] = None,
    routing: Optional[IndexRoutingStrategy] = None,
    batch_size: int = 32,
    dry_run: bool = False,
) -> int:
    """Re-ingest all episodes for a single show.

    Useful for targeted fixes when a single show's data is incorrect or
    missing, without running the full backfill.

    Args:
        show_id:    The show to re-ingest.
        storage:    Storage backend. Defaults to create_storage().
        routing:    Index routing strategy.
        batch_size: Embedding batch size.
        dry_run:    Dry-run flag.

    Returns:
        Number of episodes successfully ingested.

    Raises:
        ValueError: If show_id is not found in storage.
    """
    _storage = storage or create_storage()
    matching = [s for s in _storage.get_shows() if s.show_id == show_id]
    if not matching:
        raise ValueError(f"show_id not found in storage: {show_id!r}")

    pipeline = EmbedAndIngestPipeline(
        storage=_storage,
        routing_strategy=routing or LanguageSplitRoutingStrategy(),
        allowed_show_ids={show_id},
        batch_size=batch_size,
        dry_run=dry_run,
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
        language_distribution: Count per detected language (incl. "uncertain").
        ingest_success:        Number of documents successfully indexed.
        ingest_failed:         Number of documents that failed to index.
    """
    total = ingest_success + ingest_failed
    uncertain_rate = (
        language_distribution.get("uncertain", 0) / total if total > 0 else 0.0
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-full", action="store_true", help="Ignore cursor, process all shows")
    parser.add_argument("--show-id", type=str, help="Show ID for single-show upsert (SYNC_MODE=single)")
    args = parser.parse_args()

    setup_logging()

    mode = settings.SYNC_MODE

    if mode == "single":
        if not args.show_id:
            raise SystemExit("--show-id is required when SYNC_MODE=single")
        count = upsert_by_show_id(args.show_id, batch_size=args.batch_size, dry_run=args.dry_run)
        logger.info("upsert_complete", extra={"show_id": args.show_id, "episodes": count})
    elif mode == "backfill":
        run_backfill(batch_size=args.batch_size, dry_run=args.dry_run)
    else:
        # incremental (default) or full
        run_incremental(
            force_full=(mode == "full" or args.force_full),
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    run()
