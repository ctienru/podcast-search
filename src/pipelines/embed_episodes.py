"""Embed Episodes Pipeline

Read embedding_input files, generate embeddings, and write to a local disk cache.
Does NOT write to Elasticsearch. Designed to be run before embed_and_ingest --from-cache
so that the expensive embedding step is decoupled from ES availability.

Typical two-step workflow:
    # Step 1 (local, slow): compute vectors and cache to disk
    python -m src.pipelines.embed_episodes

    # Step 2 (fast): read cache, write to remote ES
    ES_HOST=<remote> python -m src.pipelines.embed_and_ingest --from-cache

Cache location: data/embeddings/<show_id>.json (one file per show)
Cache format:
    {
        "show_id": "<show_id>",
        "model_key": "zh" | "en",
        "model_name": "<HuggingFace model id>",
        "embedding_version": "<model_name>/<EMBEDDING_TEXT_VERSION>",
        "embedded_at": "<ISO timestamp>",
        "episodes": {
            "<episode_id>": [<float>, ...]
        }
    }

Usage:
    python -m src.pipelines.embed_episodes
    python -m src.pipelines.embed_episodes --show-ids id1 id2 id3
    python -m src.pipelines.embed_episodes --force     # re-embed even if cache exists
    python -m src.pipelines.embed_episodes --batch-size 128
"""

import argparse
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from src.config import settings
from src.embedding.backend import LocalEmbeddingBackend, MODEL_MAP
from src.types import Language
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# Map raw SQLite target_index values to Language literals (mirrors embed_and_ingest).
_TARGET_INDEX_TO_LANGUAGE: dict[str, Language] = {
    "podcast-episodes-zh-tw": "zh-tw",
    "podcast-episodes-zh-cn": "zh-cn",
    "podcast-episodes-en":    "en",
}

# Map Language → model key used in _MODEL_MAP.
_LANGUAGE_TO_MODEL_KEY: dict[Language, str] = {
    "zh-tw": "zh",
    "zh-cn": "zh",
    "en":    "en",
}

EMBEDDING_INPUT_DIR = Path("data/embedding_input/episodes")
CLEANED_EPISODES_DIR = Path("data/cleaned/episodes")


def _load_cleaned_cache(allowed_show_ids: Optional[set[str]] = None) -> Dict[str, Dict]:
    """Load cleaned episode JSON files into memory for language detection."""
    if not CLEANED_EPISODES_DIR.exists():
        logger.warning("cleaned_episodes_dir_not_found", extra={"path": str(CLEANED_EPISODES_DIR)})
        return {}

    paths = list(CLEANED_EPISODES_DIR.glob("*.json"))

    def _read(p: Path) -> Optional[Dict]:
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("cleaned_episode_load_failed", extra={"file": str(p), "error": str(e)})
            return None

    cache: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for data in pool.map(_read, paths):
            if data is None:
                continue
            episode_id = data.get("episode_id")
            show_id = data.get("show_id")
            if episode_id and (allowed_show_ids is None or show_id in allowed_show_ids):
                cache[episode_id] = data
    return cache


def _language_for_episode(episode_id: str, cleaned_cache: Dict[str, Dict]) -> Optional[Language]:
    """Derive Language from the cleaned episode cache entry."""
    ep = cleaned_cache.get(episode_id)
    if ep is None:
        return None
    return _TARGET_INDEX_TO_LANGUAGE.get(ep.get("target_index", ""))


def _model_key_for_language(lang: Language) -> str:
    return _LANGUAGE_TO_MODEL_KEY.get(lang, "en")


def _cache_path(show_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"{show_id}.json"


def _load_existing_cache(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run(
    allowed_show_ids: Optional[set[str]] = None,
    force: bool = False,
    batch_size: int = 64,
    cache_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Compute embeddings and write to disk cache.

    Args:
        allowed_show_ids: Restrict to these show IDs. None = all shows.
        force:            Re-embed even when a valid cache already exists.
        batch_size:       Encoding batch size per language group.
        cache_dir:        Cache root directory. Defaults to settings.EMBEDDING_CACHE_DIR.

    Returns:
        Stats dict with keys: written, skipped, failed, total.
    """
    _cache_dir = cache_dir or settings.EMBEDDING_CACHE_DIR
    _cache_dir.mkdir(parents=True, exist_ok=True)

    backend = LocalEmbeddingBackend()

    if not EMBEDDING_INPUT_DIR.exists():
        logger.warning("embedding_input_dir_not_found", extra={"path": str(EMBEDDING_INPUT_DIR)})
        return {"written": 0, "skipped": 0, "failed": 0, "total": 0}

    # Load cleaned cache for language detection.
    cleaned_cache = _load_cleaned_cache(allowed_show_ids)

    # Load all embedding input files; group by show_id.
    inputs_by_show: Dict[str, list] = defaultdict(list)
    total = 0
    for path in EMBEDDING_INPUT_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("embedding_input_load_failed", extra={"file": str(path), "error": str(e)})
            continue
        show_id = data.get("show_id")
        if show_id and (allowed_show_ids is None or show_id in allowed_show_ids):
            inputs_by_show[show_id].append(data)
            total += 1

    logger.info("embed_episodes_start", extra={"shows": len(inputs_by_show), "episodes": total})

    stats = {"written": 0, "skipped": 0, "failed": 0, "total": total}

    for show_id, inputs in inputs_by_show.items():
        out_path = _cache_path(show_id, _cache_dir)

        # Determine model for this show (use first episode with a known language).
        lang: Optional[Language] = None
        for inp in inputs:
            lang = _language_for_episode(inp.get("episode_id", ""), cleaned_cache)
            if lang:
                break

        if lang is None:
            logger.debug("embed_episodes_show_skipped_no_lang", extra={"show_id": show_id})
            stats["failed"] += len(inputs)
            continue

        model_key = _model_key_for_language(lang)
        model_name = MODEL_MAP[model_key]

        # Load existing cache (if valid) and find which episodes are missing.
        embedding_version = f"{model_name}/{settings.EMBEDDING_TEXT_VERSION}"
        existing_episodes: Dict[str, list] = {}
        if not force:
            existing = _load_existing_cache(out_path)
            if (
                existing
                and existing.get("model_name") == model_name
                and existing.get("embedding_version") == embedding_version
            ):
                existing_episodes = existing.get("episodes", {})

        cached_ids = set(existing_episodes)
        inputs_to_embed = [inp for inp in inputs if inp.get("episode_id") not in cached_ids]
        already_cached_count = len(inputs) - len(inputs_to_embed)

        if not inputs_to_embed:
            logger.debug("embed_episodes_cache_hit", extra={"show_id": show_id})
            stats["skipped"] += len(inputs)
            continue

        stats["skipped"] += already_cached_count

        # Group only the uncached inputs by language.
        lang_groups: Dict[Language, list] = defaultdict(list)
        for inp in inputs_to_embed:
            ep_lang = _language_for_episode(inp.get("episode_id", ""), cleaned_cache)
            if ep_lang:
                lang_groups[ep_lang].append(inp)
            else:
                logger.debug(
                    "embed_episodes_episode_skipped_no_lang",
                    extra={"episode_id": inp.get("episode_id")},
                )

        episodes_vectors: Dict[str, list] = {}

        for ep_lang, lang_inputs in lang_groups.items():
            texts = [inp["embedding_input"]["text"] for inp in lang_inputs]
            episode_ids = [inp["episode_id"] for inp in lang_inputs]

            # Encode in batches.
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                batch_ids = episode_ids[start : start + batch_size]
                vectors = backend.embed_batch(batch_texts, ep_lang)
                for ep_id, vec in zip(batch_ids, vectors):
                    episodes_vectors[ep_id] = vec

        if not episodes_vectors:
            logger.warning("embed_episodes_no_vectors", extra={"show_id": show_id})
            stats["failed"] += len(inputs_to_embed)
            continue

        # Merge new vectors into existing cache (existing entries are preserved).
        merged_episodes = {**existing_episodes, **episodes_vectors}

        cache_entry = {
            "show_id": show_id,
            "model_key": model_key,
            "model_name": model_name,
            "embedding_version": embedding_version,
            "embedded_at": datetime.now(timezone.utc).isoformat(),
            "episodes": merged_episodes,
        }

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f)
            stats["written"] += len(episodes_vectors)  # only newly computed
            logger.debug("embed_episodes_written", extra={"show_id": show_id, "count": len(episodes_vectors)})
        except Exception as e:
            logger.warning("embed_episodes_write_failed", extra={"show_id": show_id, "error": str(e)})
            stats["failed"] += len(inputs)

    logger.info("embed_episodes_complete", extra=stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed episodes and write vectors to local disk cache")
    parser.add_argument("--show-ids", nargs="+", metavar="SHOW_ID", help="Only embed these show IDs")
    parser.add_argument("--force", action="store_true", help="Re-embed even if cache exists with same model")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    setup_logging()

    show_ids_filter = set(args.show_ids) if args.show_ids else None
    run(allowed_show_ids=show_ids_filter, force=args.force, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
