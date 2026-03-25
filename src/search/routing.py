from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import IndexAlias


class IndexRoutingStrategy(ABC):
    """Determines which ES alias an episode should be written to.

    Pattern: Strategy — swap routing logic without changing the ingest pipeline.
    """

    @abstractmethod
    def get_alias(self, target_index: str) -> IndexAlias:
        """Map a raw target_index string to an ES alias name.

        Args:
            target_index: Value stored in SQLite, e.g. "podcast-episodes-zh-tw".

        Returns:
            ES alias name, e.g. "episodes-zh-tw".

        Raises:
            ValueError: If target_index cannot be mapped to a known alias.
        """
        ...


class LanguageSplitRoutingStrategy(IndexRoutingStrategy):
    """Routes episodes to language-specific indices based on target_index.

    Covers the three v2 language indices. Add entries to _ALIAS_MAP to extend.
    """

    _ALIAS_MAP: dict[str, IndexAlias] = {
        "podcast-episodes-zh-tw": "episodes-zh-tw",
        "podcast-episodes-zh-cn": "episodes-zh-cn",
        "podcast-episodes-en":    "episodes-en",
    }

    def get_alias(self, target_index: str) -> IndexAlias:
        try:
            return self._ALIAS_MAP[target_index]
        except KeyError:
            raise ValueError(f"Unknown target_index: {target_index!r}")
