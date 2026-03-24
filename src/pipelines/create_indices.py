import logging
from typing import Optional

from src.es.mapping_loader import MappingLoader
from src.services.es_service import ElasticsearchService
from src.config.settings import (
    MAPPINGS_DIR,
    INDEX_VERSION,
    REINDEX,
    ALLOW_DELETE_BASE_INDEX,
    ENABLE_LANGUAGE_SPLIT,
)
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class CreateIndicesPipeline:
    """
    Create versioned Elasticsearch indices, reindex data,
    and switch aliases atomically.
    """

    def __init__(
        self,
        es_service: Optional[ElasticsearchService] = None,
        mapping_loader: Optional[MappingLoader] = None,
        index_version: Optional[int] = None,
        reindex: Optional[bool] = None,
        allow_delete_base_index: Optional[bool] = None,
        enable_language_split: Optional[bool] = None,
    ) -> None:
        self.es = es_service or ElasticsearchService()
        self.mappings = mapping_loader or MappingLoader(MAPPINGS_DIR)
        self.index_version = index_version if index_version is not None else INDEX_VERSION
        self.reindex = reindex if reindex is not None else REINDEX
        self.allow_delete_base_index = (
            allow_delete_base_index if allow_delete_base_index is not None else ALLOW_DELETE_BASE_INDEX
        )
        self.enable_language_split = (
            enable_language_split if enable_language_split is not None else ENABLE_LANGUAGE_SPLIT
        )
    # ---------- helpers ----------

    def _versioned_index(self, base: str) -> str:
        return f"{base}_v{self.index_version}"

    def ensure_alias_name_is_free(self, alias: str) -> None:
        """
        Ensure alias name is not occupied by a concrete index.
        Alias itself is allowed and will be handled in switch_alias.
        """

        # Case 1: alias exists → OK, will be removed in switch_alias
        if self.es.alias_exists(alias):
            logger.info(
                "alias_already_exists",
                extra={"alias": alias},
            )
            return

        # Case 2: concrete index exists with same name
        if self.es.index_exists(alias):
            if not self.allow_delete_base_index:
                raise RuntimeError(
                    f"alias_name_conflict: index '{alias}' exists. "
                    f"Set ALLOW_DELETE_BASE_INDEX=true to auto-delete."
                )

            logger.warning(
                "delete_conflicting_index",
                extra={"index": alias},
            )
            self.es.delete_index(alias)

    # ---------- core steps ----------

    def create_versioned_index(self, base_index: str, mapping_key: Optional[str] = None) -> str:
        """Create a versioned index, loading its mapping from mappings/.

        Args:
            base_index: Base name of the index, e.g. "podcast-episodes-zh-tw".
            mapping_key: Key passed to MappingLoader.load(); defaults to base_index.
                Used when the mapping filename differs from the index base name,
                e.g. base_index="podcast-episodes-zh-tw", mapping_key="episodes-zh-tw".
        """
        index_name = self._versioned_index(base_index)

        if self.es.index_exists(index_name):
            logger.info(
                "index_version_exists",
                extra={"index": index_name},
            )
            return index_name

        body = self.mappings.load(mapping_key or base_index)

        logger.info(
            "create_index_start",
            extra={
                "index": index_name,
                "base_index": base_index,
                "version": self.index_version,
            },
        )

        self.es.create_index(index_name, body)

        logger.info(
            "create_index_success",
            extra={"index": index_name},
        )

        return index_name

    def reindex_if_needed(self, source: str, dest: str) -> None:
        if not self.reindex:
            logger.info(
                "reindex_skipped",
                extra={"source": source, "dest": dest},
            )
            return

        if not self.es.index_exists(source):
            logger.info(
                "reindex_source_missing",
                extra={"source": source},
            )
            return

        logger.info(
            "reindex_start",
            extra={"source": source, "dest": dest},
        )

        self.es.reindex(source, dest)

        logger.info(
            "reindex_completed",
            extra={"source": source, "dest": dest},
        )

    def switch_alias(self, alias: str, new_index: str) -> None:
        logger.info(
            "alias_switch_start",
            extra={"alias": alias, "new_index": new_index},
        )

        # safety check
        self.ensure_alias_name_is_free(alias)

        actions = []

        if self.es.alias_exists(alias):
            actions.append(
                {"remove": {"alias": alias, "index": "*"}}
            )

        actions.append(
            {"add": {"alias": alias, "index": new_index}}
        )

        self.es.update_aliases(actions)

        logger.info(
            "alias_switch_completed",
            extra={"alias": alias, "new_index": new_index},
        )

    # ---------- orchestration ----------

    def run_for_index(self, base_index: str) -> None:
        new_index = self.create_versioned_index(base_index)

        # Find the old versioned index to reindex from
        old_index = None
        if self.index_version > 1:
            old_version = self.index_version - 1
            old_index = f"{base_index}_v{old_version}"

        # If old versioned index doesn't exist, try the alias (which might point to an older version)
        if old_index and not self.es.index_exists(old_index):
            if self.es.alias_exists(base_index):
                old_index = base_index
            else:
                old_index = None

        if old_index:
            self.reindex_if_needed(old_index, new_index)

        self.switch_alias(base_index, new_index)

    def _run_language_split(self) -> None:
        """v2: create three language-specific episode indices plus the shows index."""
        # shows: base_name uses podcast- prefix to match infra naming convention
        shows_base = "podcast-shows"
        shows_alias = "shows"
        new_shows_index = self.create_versioned_index(shows_base, mapping_key="shows")

        old_shows: Optional[str] = None
        if self.index_version > 1:
            old_shows = f"{shows_base}_v{self.index_version - 1}"
            if not self.es.index_exists(old_shows):
                old_shows = shows_alias if self.es.alias_exists(shows_alias) else None

        if old_shows:
            self.reindex_if_needed(old_shows, new_shows_index)

        self.switch_alias(shows_alias, new_shows_index)

        for lang in ("zh-tw", "zh-cn", "en"):
            base_name = f"podcast-episodes-{lang}"
            alias = f"episodes-{lang}"
            new_index = self.create_versioned_index(base_name, mapping_key=f"episodes-{lang}")

            old_index: Optional[str] = None
            if self.index_version > 1:
                old_index = f"{base_name}_v{self.index_version - 1}"
                if not self.es.index_exists(old_index):
                    # Fall back to alias (may point to an older versioned index)
                    old_index = alias if self.es.alias_exists(alias) else None

            if old_index:
                self.reindex_if_needed(old_index, new_index)

            self.switch_alias(alias, new_index)

    def run(self) -> None:
        if self.enable_language_split:
            self._run_language_split()
        else:
            for base in ["shows", "episodes"]:
                self.run_for_index(base)


def run() -> None:
    setup_logging()
    CreateIndicesPipeline().run()


if __name__ == "__main__":
    run()