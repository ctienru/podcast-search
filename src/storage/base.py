from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

from src.types import Language, Show


class StorageBase(ABC):
    """Abstract base for all storage backends.

    Read-only: write operations belong to podcast-crawler, not podcast-search.
    Implement this interface to add a new storage backend without touching
    pipeline code.
    """

    @abstractmethod
    def get_shows(self, language: Language | None = None) -> Iterator[Show]:
        """Yield shows from the storage backend.

        Args:
            language: If provided, yield only shows matching this language.
                      If None, yield all shows with a non-null target_index.

        Yields:
            Show dataclass instances.
        """
        ...

    @abstractmethod
    def get_shows_updated_since(
        self,
        since: str,
        language: Language | None = None,
    ) -> Iterator[Show]:
        """Yield only shows updated after the given timestamp.

        Used for incremental ingest: pass the cursor's last_ingest_at so only
        new or re-crawled shows are processed, not the full dataset.

        Args:
            since:    ISO 8601 UTC timestamp. Yields shows where updated_at > since.
                      Empty string means no filter (same as get_shows).
            language: Optional language filter (same semantics as get_shows).

        Yields:
            Show dataclass instances.
        """
        ...
