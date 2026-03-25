"""Phase 2 migration: reindex all episodes into three language-specific aliases.

Steps
-----
1. Read all shows from SQLite (with target_index).
2. For each show, read episodes from RSS XML and apply show's target_index.
3. Group episodes by target_index; bulk ingest to the three new aliases.
4. Verify count alignment: sum(new indices) >= old index count.
5. Print alias switch commands for manual review and execution.
6. Print deprecation notice for the old episodes index.

Prerequisites
-------------
- ENABLE_LANGUAGE_SPLIT=true
- crawler v2 has run and written language detection results to SQLite
- Three new versioned indices exist (run create_indices.py first)
- Offline regression gate has passed (NDCG >= v1 baseline)

Usage
-----
    python scripts/migrate_reindex.py \\
        --raw-rss-dir ../podcast-crawler/data/raw/rss \\
        --dry-run

    # When ready to ingest:
    python scripts/migrate_reindex.py \\
        --raw-rss-dir ../podcast-crawler/data/raw/rss
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from src.cleaning.rss_parser import RSSParser
from src.config.settings import (
    ES_HOST,
    ES_API_KEY,
    INDEX_VERSION,
    SQLITE_PATH,
)
from src.search.routing import LanguageSplitRoutingStrategy
from src.services.es_service import ElasticsearchService
from src.storage.sqlite import SQLiteStorage
from src.types import IndexAlias
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

_LANG_ALIASES: list[IndexAlias] = ["episodes-zh-tw", "episodes-zh-cn", "episodes-en"]
_OLD_ALIAS = "episodes"


def _iter_episode_actions(
    sqlite_path: Path,
    raw_rss_dir: Path,
    router: LanguageSplitRoutingStrategy,
) -> Iterator[Dict]:
    """Yield ES bulk actions for all episodes, routed to the correct alias.

    Args:
        sqlite_path: Path to the crawler SQLite database.
        raw_rss_dir: Directory containing RSS XML files.
        router:      Routing strategy mapping target_index → ES alias.
    """
    storage = SQLiteStorage(sqlite_path)
    parser = RSSParser()

    for show in storage.get_shows():
        xml_path = raw_rss_dir / f"{show.show_id}.xml"
        if not xml_path.exists():
            logger.warning(
                "rss_xml_not_found",
                extra={"show_id": show.show_id, "path": str(xml_path)},
            )
            continue

        try:
            alias = router.get_alias(show.target_index)
        except ValueError:
            logger.warning(
                "show_routing_failed",
                extra={"show_id": show.show_id, "target_index": show.target_index},
            )
            continue

        try:
            _, episodes = parser.parse_file(xml_path)
        except Exception as exc:
            logger.warning(
                "rss_parse_failed",
                extra={"show_id": show.show_id, "error": str(exc)},
            )
            continue

        for ep in episodes:
            now = datetime.now(timezone.utc).isoformat()
            yield {
                "_index": alias,
                "_id": ep.episode_id,
                "_source": {
                    "episode_id": ep.episode_id,
                    "title": ep.title,
                    "description": ep.description or "",
                    "language": show.language_detected,
                    "show": {
                        "show_id": show.show_id,
                        "title": show.title,
                        "publisher": show.author,
                    },
                    "audio": {"url": ep.audio_url},
                    "created_at": now,
                    "updated_at": now,
                },
            }


def _count_index(es: Elasticsearch, index: str) -> int:
    """Return document count for an index/alias, or -1 if not found."""
    try:
        result = es.count(index=index)
        return result["count"]
    except Exception:
        return -1


def _verify_counts(es: Elasticsearch) -> bool:
    """Check that sum of new indices >= old index count.

    Returns True if the verification passes.
    """
    old_count = _count_index(es, _OLD_ALIAS)
    new_counts = {alias: _count_index(es, alias) for alias in _LANG_ALIASES}
    new_total = sum(c for c in new_counts.values() if c >= 0)

    logger.info(
        "count_verification",
        extra={
            "old_index": old_count,
            "new_indices": new_counts,
            "new_total": new_total,
        },
    )

    print("\n── Count Verification ──────────────────────────────────")
    print(f"  {_OLD_ALIAS} (old):  {old_count}")
    for alias, count in new_counts.items():
        print(f"  {alias}:  {count}")
    print(f"  Sum of new indices: {new_total}")

    if old_count < 0:
        print("  ⚠  Old index not found — skipping comparison.")
        return True

    if new_total >= old_count:
        print(f"  ✓  new_total ({new_total}) >= old ({old_count})")
        return True
    else:
        print(f"  ✗  new_total ({new_total}) < old ({old_count}) — investigation required.")
        return False


def _print_alias_switch_commands(index_version: int) -> None:
    """Print the ES alias switch commands for manual review."""
    print("\n── Alias Switch Commands (review, then execute manually) ──")
    print("POST /_aliases")
    print("{")
    print('  "actions": [')
    for lang in ("zh-tw", "zh-cn", "en"):
        alias = f"episodes-{lang}"
        index = f"podcast-episodes-{lang}_v{index_version}"
        print(f'    {{ "remove": {{ "index": "*", "alias": "{alias}" }} }},')
        print(f'    {{ "add":    {{ "index": "{index}", "alias": "{alias}" }} }},')
    print("  ]")
    print("}")


def run(
    raw_rss_dir: Path,
    dry_run: bool = False,
    chunk_size: int = 500,
) -> None:
    """Run the Phase 2 migration.

    Args:
        raw_rss_dir: Directory containing RSS XML files from podcast-crawler.
        dry_run:     If True, count episodes but do not ingest.
        chunk_size:  Bulk ingest chunk size.
    """
    setup_logging()
    router = LanguageSplitRoutingStrategy()
    es_service = ElasticsearchService()

    actions = _iter_episode_actions(SQLITE_PATH, raw_rss_dir, router)

    # Count per alias for reporting
    alias_counts: Dict[str, int] = defaultdict(int)

    if dry_run:
        logger.info("dry_run_mode_counting_episodes")
        print("Dry run — counting episodes without ingesting...")
        for action in actions:
            alias_counts[action["_index"]] += 1
        print("\nEpisode counts by alias:")
        for alias, count in sorted(alias_counts.items()):
            print(f"  {alias}: {count}")
        print(f"  Total: {sum(alias_counts.values())}")
        return

    # Step 3: bulk ingest
    logger.info("migration_start")
    success = 0
    failed: List = []

    def _tracked_actions() -> Iterator[Dict]:
        for action in _iter_episode_actions(SQLITE_PATH, raw_rss_dir, router):
            alias_counts[action["_index"]] += 1
            yield action

    for ok, item in streaming_bulk(
        es_service.client,
        _tracked_actions(),
        chunk_size=chunk_size,
        raise_on_error=False,
    ):
        if ok:
            success += 1
            if success % 1000 == 0:
                logger.info("migration_progress", extra={"ingested": success})
        else:
            failed.append(item)

    logger.info(
        "migration_complete",
        extra={"success": success, "failed": len(failed), "by_alias": dict(alias_counts)},
    )
    print(f"\nIngested: {success}  Failed: {len(failed)}")
    for alias, count in sorted(alias_counts.items()):
        print(f"  {alias}: {count}")

    # Step 4: verify counts
    passed = _verify_counts(es_service.client)

    if not passed:
        print("\n✗ Count verification failed. Do NOT switch aliases until investigated.")
        sys.exit(1)

    # Step 5: print alias switch commands
    _print_alias_switch_commands(INDEX_VERSION)

    # Step 6: deprecation notice
    print(f"\n── Deprecation Notice ──────────────────────────────────")
    print(f"  After aliases are switched, mark '{_OLD_ALIAS}' as deprecated.")
    print(f"  Retain for 7 days, then delete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 migration: reindex to language-split indices")
    parser.add_argument(
        "--raw-rss-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1].parent / "podcast-crawler" / "data" / "raw" / "rss",
        help="Path to RSS XML directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count episodes without ingesting",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Bulk ingest chunk size (default: 500)",
    )
    args = parser.parse_args()

    run(
        raw_rss_dir=args.raw_rss_dir,
        dry_run=args.dry_run,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
