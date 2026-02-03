"""
Embed and Ingest Shows Pipeline

Read embedding_input files for shows, generate embeddings, and write to Elasticsearch.

Data sources:
- embedding_input: data/embedding_input/shows/*.json (text for embedding)
- shows: canonical show data from storage

Usage:
    python -m src.pipelines.embed_and_ingest_shows
    python -m src.pipelines.embed_and_ingest_shows --batch-size 64
    python -m src.pipelines.embed_and_ingest_shows --dry-run
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

logger = logging.getLogger(__name__)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)
logging.getLogger("elasticsearch").setLevel(logging.WARNING)


class EmbedAndIngestShowsPipeline:
    """
    Pipeline to embed show texts and ingest into Elasticsearch.

    Data flow:
        embedding_input/*.json + canonical show data → EmbeddingEncoder → ES shows index

    The pipeline:
    1. Reads pre-built embedding_input files (title-weighted + truncated text)
    2. Loads canonical show data for metadata
    3. Generates embeddings using sentence-transformers
    4. Merges embedding into ES document
    5. Bulk indexes into ES
    """

    INDEX_ALIAS = "shows"
    EMBEDDING_INPUT_DIR = Path("data/embedding_input/shows")

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

        # Cache for show data
        self._show_cache: Dict[str, Dict] = {}

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

    def _get_show_data(self, show_id: str) -> Optional[Dict]:
        """Get show data from cache."""
        return self._show_cache.get(show_id)

    def list_embedding_input_files(self) -> list[Path]:
        """List all embedding input JSON files for shows."""
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
        Build ES document from embedding_input, show data, and embedding vector.
        """
        show_id = embedding_input["show_id"]

        # Load show data from cache
        show_data = self._get_show_data(show_id)
        if not show_data:
            logger.warning(
                "show_not_found",
                extra={"show_id": show_id},
            )
            return None

        # Extract data from canonical show
        external_urls = show_data.get("external_urls") or {}
        image = show_data.get("image") or {}
        episode_stats = show_data.get("episode_stats") or {}
        provider = show_data.get("provider", "apple_podcasts")

        now = datetime.now(timezone.utc).isoformat()

        return {
            "_index": self.INDEX_ALIAS,
            "_id": show_id,
            "_source": {
                "show_id": show_id,

                # External IDs
                "external_ids": {
                    provider: show_data.get("external_id"),
                },

                # External URLs
                "external_urls": {
                    provider: external_urls.get(provider),
                },

                # Content
                "title": show_data.get("title"),
                "publisher": show_data.get("author"),
                "description": show_data.get("description"),

                # Embedding
                "embedding": embedding_vector,

                "language": show_data.get("language"),

                # Episode stats
                "episode_count": episode_stats.get("episode_count"),
                "last_episode_at": episode_stats.get("last_episode_at"),

                # Ranking
                "popularity_score": None,

                # Media
                "image_url": image.get("url"),

                # Timestamps
                "created_at": show_data.get("created_at") or now,
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
        Run the embed and ingest pipeline for shows.

        Returns stats dict.
        """
        start_time = datetime.now(timezone.utc)

        # Load show cache
        self._load_show_cache()

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
                        "show_id": inp["show_id"],
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
    parser = argparse.ArgumentParser(description="Embed and ingest shows to ES")
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

    pipeline = EmbedAndIngestShowsPipeline(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    pipeline.run()


if __name__ == "__main__":
    run()
