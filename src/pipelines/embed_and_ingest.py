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
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

from elasticsearch.helpers import streaming_bulk

from src.embedding.encoder import EmbeddingEncoder
from src.services.es_service import ElasticsearchService
from src.storage import storage
from src.utils.logging import setup_logging
from src.utils.parsers import normalize_language, parse_duration, parse_pub_date

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class EmbedAndIngestPipeline:
    """
    Pipeline to embed episode texts and ingest into Elasticsearch.

    Data flow:
        embedding_input/*.json + cleaned/*.json → EmbeddingEncoder → ES episodes index

    The pipeline:
    1. Reads pre-built embedding_input files (title-weighted + truncated text)
    2. Loads corresponding cleaned episode data for metadata
    3. Generates embeddings using sentence-transformers
    4. Merges embedding into ES document
    5. Bulk indexes into ES
    """

    INDEX_ALIAS = "episodes"
    EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")
    CLEANED_EPISODES_DIR = Path("data/cleaned/episodes")
    CLEANED_SHOWS_DIR = Path("data/cleaned/shows")

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
        encoder: Optional[EmbeddingEncoder] = None,
        batch_size: int = 32,
        dry_run: bool = False,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        self.batch_size = batch_size
        self.dry_run = dry_run

        # Lazy load encoder (it's heavy)
        self._encoder = encoder

        # Caches
        self._show_cache: Dict[str, Dict] = {}
        self._cleaned_episode_cache: Dict[str, Dict] = {}

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
        """Pre-load all show data from storage."""
        loaded = 0
        for show_id in storage.list_show_ids():
            try:
                show_data = storage.load_show(show_id)
                if show_data:
                    self._show_cache[show_id] = show_data
                    loaded += 1
            except Exception as e:
                logger.warning(
                    "show_cache_load_failed",
                    extra={"show_id": show_id, "error": str(e)},
                )

        logger.info(
            "show_cache_loaded",
            extra={"count": loaded},
        )

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
                    if episode_id:
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
        """Load all embedding input files."""
        files = self.list_embedding_input_files()
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    yield json.load(fp)
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

        now = datetime.now(timezone.utc).isoformat()

        return {
            "_index": self.INDEX_ALIAS,
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

        return stats


def run() -> None:
    parser = argparse.ArgumentParser(description="Embed and ingest episodes to ES")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding encoding (default: 32)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - don't actually ingest",
    )
    args = parser.parse_args()

    setup_logging()

    pipeline = EmbedAndIngestPipeline(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    pipeline.run()


if __name__ == "__main__":
    run()
