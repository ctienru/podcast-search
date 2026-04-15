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

from sqlite_utils import Database
from tqdm import tqdm

from src.config import settings
from src.embedding.backend import LocalEmbeddingBackend
from src.pipelines.embedding_catalog import MODEL_MAP
from src.storage.episode_status import EpisodeStatusRepository
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


def _load_cleaned_cache(allowed_show_ids: Optional[set[str]] = None,
                        db: Optional[Database] = None) -> Dict[str, Dict]:
    """Load cleaned episode JSON files into memory for language detection.

    When allowed_show_ids is provided and db is available, episode IDs are
    fetched from SQLite first so only the relevant files are read — avoids
    scanning the entire cleaned cache directory.
    """
    if not CLEANED_EPISODES_DIR.exists():
        logger.warning("cleaned_episodes_dir_not_found", extra={"path": str(CLEANED_EPISODES_DIR)})
        return {}

    if allowed_show_ids and db is not None:
        # Fast path: resolve episode IDs from SQLite, build paths directly.
        placeholders = ",".join("?" * len(allowed_show_ids))
        rows = db.execute(
            f"SELECT episode_id FROM episodes WHERE show_id IN ({placeholders})",
            list(allowed_show_ids),
        ).fetchall()
        paths = [
            CLEANED_EPISODES_DIR / f"{row[0].replace(':', '_')}.json"
            for row in rows
        ]
        paths = [p for p in paths if p.exists()]
    else:
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
        for data in tqdm(pool.map(_read, paths), total=len(paths),
                         desc="Loading cleaned cache", unit="ep", leave=False):
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


def _load_show_target_index_map(
    db: Optional[Database] = None,
    allowed_show_ids: Optional[set[str]] = None,
) -> Dict[str, Optional[str]]:
    """Load show_id -> target_index from SQLite when available.

    This lets embed_episodes align its eligibility rules with the rest of the
    v2 pipeline: shows with target_index=NULL are excluded from embedding
    coverage rather than being counted as failures.
    """
    if db is None or "shows" not in db.table_names():
        return {}

    if allowed_show_ids:
        placeholders = ",".join("?" * len(allowed_show_ids))
        rows = db.execute(
            f"SELECT show_id, target_index FROM shows WHERE show_id IN ({placeholders})",
            list(allowed_show_ids),
        ).fetchall()
    else:
        rows = db.execute("SELECT show_id, target_index FROM shows").fetchall()

    return {row[0]: row[1] for row in rows}


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
    db: Optional[Database] = None,
) -> Dict[str, int]:
    """Compute embeddings and write to disk cache.

    Args:
        allowed_show_ids: Restrict to these show IDs. None = all shows.
        force:            Re-embed even when a valid cache already exists.
        batch_size:       Encoding batch size per language group.
        cache_dir:        Cache root directory. Defaults to settings.EMBEDDING_CACHE_DIR.
        db:               Optional sqlite-utils Database for writing embedding_status='done'
                          back to episodes table. Pass None to skip DB writeback.

    Returns:
        Stats dict with keys: written, skipped, failed, excluded, total.
    """
    _cache_dir = cache_dir or settings.EMBEDDING_CACHE_DIR
    _cache_dir.mkdir(parents=True, exist_ok=True)

    ep_repo = EpisodeStatusRepository(db) if db is not None else None
    backend = LocalEmbeddingBackend()
    show_target_index_map = _load_show_target_index_map(db, allowed_show_ids)

    if not EMBEDDING_INPUT_DIR.exists():
        logger.warning("embedding_input_dir_not_found", extra={"path": str(EMBEDDING_INPUT_DIR)})
        return {"written": 0, "skipped": 0, "failed": 0, "excluded": 0, "total": 0}

    # Load cleaned cache for language detection.
    # Pass db so the fast path (SQLite episode ID lookup) can be used when
    # allowed_show_ids is set, avoiding a full directory scan.
    cleaned_cache = _load_cleaned_cache(allowed_show_ids, db=db)

    # Load all embedding input files; group by show_id.
    input_paths = list(EMBEDDING_INPUT_DIR.glob("*.json"))
    inputs_by_show: Dict[str, list] = defaultdict(list)
    total = 0
    excluded = 0
    for path in tqdm(input_paths, desc="Loading embedding inputs", unit="ep", leave=False):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("embedding_input_load_failed", extra={"file": str(path), "error": str(e)})
            continue
        show_id = data.get("show_id")
        if show_id and (allowed_show_ids is None or show_id in allowed_show_ids):
            if show_id in show_target_index_map and show_target_index_map[show_id] is None:
                excluded += 1
                continue
            inputs_by_show[show_id].append(data)
            total += 1

    logger.info(
        "embed_episodes_start",
        extra={"shows": len(inputs_by_show), "episodes": total, "excluded": excluded},
    )

    # Warm up MPS JIT before the main loop so tqdm's time estimate is not
    # thrown off by the first-call Metal compilation overhead.
    print("Warming up embedding model...", flush=True)
    backend.embed_batch(["warmup"], "zh-tw")
    backend.embed_batch(["warmup"], "en")
    print("  Ready.", flush=True)

    stats = {"written": 0, "skipped": 0, "failed": 0, "excluded": excluded, "total": total}

    with tqdm(total=total, desc="Embedding episodes", unit="ep") as pbar:
        for show_id, inputs in inputs_by_show.items():
            show_episode_count = len(inputs)
            out_path = _cache_path(show_id, _cache_dir)

            # Prefer SQLite show.target_index: it is the source of truth for v2
            # routing and avoids false failures when older cleaned JSON files do
            # not carry target_index yet.
            raw_show_target = show_target_index_map.get(show_id)
            lang: Optional[Language] = _TARGET_INDEX_TO_LANGUAGE.get(raw_show_target or "")
            if lang is None:
                for inp in inputs:
                    lang = _language_for_episode(inp.get("episode_id", ""), cleaned_cache)
                    if lang:
                        break

            if lang is None:
                logger.debug("embed_episodes_show_skipped_no_lang", extra={"show_id": show_id})
                stats["failed"] += len(inputs)
                pbar.update(show_episode_count)
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
                pbar.update(show_episode_count)
                continue

            stats["skipped"] += already_cached_count

            # Group only the uncached inputs by language.
            lang_groups: Dict[Language, list] = defaultdict(list)
            missing_lang_count = 0
            if raw_show_target:
                lang_groups[lang] = list(inputs_to_embed)
            else:
                for inp in inputs_to_embed:
                    ep_lang = _language_for_episode(inp.get("episode_id", ""), cleaned_cache)
                    if ep_lang:
                        lang_groups[ep_lang].append(inp)
                    else:
                        missing_lang_count += 1
                        logger.debug(
                            "embed_episodes_episode_skipped_no_lang",
                            extra={"episode_id": inp.get("episode_id")},
                        )

            if missing_lang_count:
                stats["failed"] += missing_lang_count

            episodes_vectors: Dict[str, list] = {}
            expected_vectors = sum(len(group) for group in lang_groups.values())

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
                stats["failed"] += expected_vectors
                pbar.update(show_episode_count)
                continue

            missing_vectors = expected_vectors - len(episodes_vectors)
            if missing_vectors > 0:
                stats["failed"] += missing_vectors

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
                if ep_repo is not None and episodes_vectors:
                    ep_repo.mark_embedded_batch(
                        list(episodes_vectors.keys()),
                        model=model_name,
                        version=embedding_version,
                        embedded_at=cache_entry["embedded_at"],
                    )
            except Exception as e:
                logger.warning("embed_episodes_write_failed", extra={"show_id": show_id, "error": str(e)})
                stats["failed"] += len(episodes_vectors)

            pbar.set_postfix(
                written=stats["written"],
                skipped=stats["skipped"],
                failed=stats["failed"],
                excluded=stats["excluded"],
            )
            pbar.update(show_episode_count)

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
    _db = Database(settings.SQLITE_PATH)
    run(allowed_show_ids=show_ids_filter, force=args.force, batch_size=args.batch_size, db=_db)


if __name__ == "__main__":
    main()
