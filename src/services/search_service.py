"""
Search Service

Supports four search modes:
1. BM25 - Pure text search (fuzzy matching)
2. kNN - Pure semantic search (embedding similarity)
3. Hybrid - BM25 + kNN + RRF fusion (recommended)
4. Exact - Phrase matching (match_phrase for precise results)

Usage:
    from src.services.search_service import SearchService

    service = SearchService()

    # BM25 search
    results = service.search_bm25("podcast about AI", size=10)

    # kNN semantic search
    results = service.search_knn("podcast about AI", size=10)

    # Hybrid search (recommended)
    results = service.search_hybrid("podcast about AI", size=10)

    # Exact phrase matching
    results = service.search_exact("AI podcast", size=10)
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from src.embedding.backend import EmbeddingBackend
from src.embedding.factory import create_backend
from src.es.client import get_es_client
from src.types import Language

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    BM25 = "bm25"
    KNN = "knn"
    HYBRID = "hybrid"
    EXACT = "exact"


@dataclass
class SearchResult:
    """Single search result."""
    episode_id: str
    title: str
    description: Optional[str]
    score: float
    show_title: Optional[str] = None
    show_id: Optional[str] = None
    published_at: Optional[str] = None
    duration_sec: Optional[int] = None


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[SearchResult]
    total: int
    took_ms: int
    mode: SearchMode


class SearchService:
    """
    Elasticsearch search service supporting BM25, kNN, and Hybrid search.

    Hybrid search uses Reciprocal Rank Fusion (RRF) to combine:
    - BM25: Text matching on title and description
    - kNN: Semantic similarity on embedding vectors
    """

    def _get_target_index(self, language: str) -> str:
        from src.config import settings
        if not settings.ENABLE_LANGUAGE_SPLIT:
            return "episodes"
        if language == "zh-cn":
            return settings.INDEX_ZH_CN
        if language == "en":
            return settings.INDEX_EN
        return settings.INDEX_ZH_TW

    # RRF parameters
    RRF_WINDOW_SIZE = 100
    RRF_RANK_CONSTANT = 60

    # Diversity: max results from the same show in a single response
    MAX_PER_SHOW = 3

    # kNN parameters
    KNN_NUM_CANDIDATES = 100

    def __init__(
        self,
        encoder: Optional[EmbeddingBackend] = None,
    ):
        self.client = get_es_client()
        self._encoder = encoder

    @property
    def encoder(self) -> EmbeddingBackend:
        """Lazy load embedding backend."""
        if self._encoder is None:
            logger.info("loading_encoder_for_search")
            self._encoder = create_backend()
        return self._encoder

    def _build_bm25_query(
        self,
        query: str,
        evaluation_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Build BM25 query for text matching.

        Args:
            query: Search query text
            evaluation_mode: If True, use evaluation-safe query
                (no time_decay, no language filter, no match_phrase boost)
        """
        if evaluation_mode:
            # Evaluation-safe: no match_phrase boost, no filters
            return {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "title.chinese^3", "description", "description.chinese"],
                                "type": "best_fields",
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }

        # Production query with match_phrase boost
        return {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["title^3", "description"],
                            "type": "best_fields",
                        }
                    },
                    {
                        "match_phrase": {
                            "title": {
                                "query": query,
                                "boost": 2,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _build_exact_query(self, query: str) -> Dict[str, Any]:
        """Build exact match query using match_phrase for precise phrase matching."""
        return {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "title": {
                                "query": query,
                                "boost": 3,
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "title.chinese": {
                                "query": query,
                                "boost": 3,
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "description": {
                                "query": query,
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "description.chinese": {
                                "query": query,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

    def _build_knn_clause(
        self,
        query_vector: List[float],
        k: int,
    ) -> Dict[str, Any]:
        """Build kNN clause for semantic search."""
        return {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": self.KNN_NUM_CANDIDATES,
        }

    def _parse_hits(self, hits: List[Dict]) -> List[SearchResult]:
        """Parse ES hits into SearchResult objects."""
        results = []
        for hit in hits:
            source = hit.get("_source", {})
            show = source.get("show", {})

            results.append(SearchResult(
                episode_id=source.get("episode_id", hit.get("_id")),
                title=source.get("title", ""),
                description=source.get("description"),
                score=hit.get("_score", 0.0),
                show_title=show.get("title"),
                show_id=show.get("show_id"),
                published_at=source.get("published_at"),
                duration_sec=source.get("duration_sec"),
            ))

        return results

    def search_bm25(
        self,
        query: str,
        size: int = 10,
        evaluation_mode: bool = False,
        language: Language = "zh-tw",
    ) -> SearchResponse:
        """
        Pure BM25 text search.

        Args:
            query: Search query text
            size: Number of results to return
            evaluation_mode: If True, use evaluation-safe query
                (no time_decay, no language filter, no match_phrase boost)
            language: Target index language for routing.

        Returns:
            SearchResponse with results
        """
        body = {
            "query": self._build_bm25_query(query, evaluation_mode=evaluation_mode),
            "size": size,
            "_source": [
                "episode_id", "title", "description",
                "show", "published_at", "duration_sec"
            ],
        }

        response = self.client.search(index=self._get_target_index(language), body=body)

        hits = response.get("hits", {})
        results = self._parse_hits(hits.get("hits", []))
        total = hits.get("total", {}).get("value", 0)
        took = response.get("took", 0)

        logger.info(
            "search_bm25",
            extra={
                "query": query,
                "results": len(results),
                "total": total,
                "took_ms": took,
                "evaluation_mode": evaluation_mode,
            },
        )

        return SearchResponse(
            results=results,
            total=total,
            took_ms=took,
            mode=SearchMode.BM25,
        )

    def search_exact(
        self,
        query: str,
        size: int = 10,
    ) -> SearchResponse:
        """
        Exact phrase match search using match_phrase.

        This mode is designed for users who want to find exact phrase matches
        rather than fuzzy text matching. Useful for finding specific podcast
        titles or exact quotes.

        Args:
            query: Search query text (will be matched as exact phrase)
            size: Number of results to return

        Returns:
            SearchResponse with results
        """
        body = {
            "query": self._build_exact_query(query),
            "size": size,
            "_source": [
                "episode_id", "title", "description",
                "show", "published_at", "duration_sec"
            ],
        }

        response = self.client.search(index=self._get_target_index(language), body=body)

        hits = response.get("hits", {})
        results = self._parse_hits(hits.get("hits", []))
        total = hits.get("total", {}).get("value", 0)
        took = response.get("took", 0)

        logger.info(
            "search_exact",
            extra={
                "query": query,
                "results": len(results),
                "total": total,
                "took_ms": took,
            },
        )

        return SearchResponse(
            results=results,
            total=total,
            took_ms=took,
            mode=SearchMode.EXACT,
        )

    def search_knn(
        self,
        query: str,
        size: int = 10,
        language: Language = "zh-tw",
    ) -> SearchResponse:
        """
        Pure kNN semantic search.

        Args:
            query: Search query text (will be encoded to vector)
            size: Number of results to return
            language: Language of the query. Must match the target index language
                so the correct embedding model and vector space are used.
                Passing the wrong language silently returns wrong results.

        Returns:
            SearchResponse with results
        """
        query_vector = self.encoder.embed(query, language=language)

        body = {
            "knn": self._build_knn_clause(query_vector, k=size),
            "size": size,
            "_source": [
                "episode_id", "title", "description",
                "show", "published_at", "duration_sec"
            ],
        }

        response = self.client.search(index=self._get_target_index(language), body=body)

        hits = response.get("hits", {})
        results = self._parse_hits(hits.get("hits", []))
        total = hits.get("total", {}).get("value", 0)
        took = response.get("took", 0)

        logger.info(
            "search_knn",
            extra={
                "query": query,
                "results": len(results),
                "total": total,
                "took_ms": took,
            },
        )

        return SearchResponse(
            results=results,
            total=total,
            took_ms=took,
            mode=SearchMode.KNN,
        )

    def _compute_rrf_scores(
        self,
        bm25_results: List[SearchResult],
        knn_results: List[SearchResult],
        rank_constant: int = 60,
    ) -> Dict[str, float]:
        """
        Compute RRF scores manually.

        RRF score = sum(1 / (rank_constant + rank_i))

        Args:
            bm25_results: Results from BM25 search
            knn_results: Results from kNN search
            rank_constant: RRF constant k (default: 60)

        Returns:
            Dict mapping episode_id to RRF score
        """
        rrf_scores: Dict[str, float] = {}

        # Add BM25 contributions
        for rank, result in enumerate(bm25_results, start=1):
            episode_id = result.episode_id
            rrf_scores[episode_id] = rrf_scores.get(episode_id, 0) + 1 / (rank_constant + rank)

        # Add kNN contributions
        for rank, result in enumerate(knn_results, start=1):
            episode_id = result.episode_id
            rrf_scores[episode_id] = rrf_scores.get(episode_id, 0) + 1 / (rank_constant + rank)

        return rrf_scores

    def search_hybrid(
        self,
        query: str,
        size: int = 10,
        rrf_window_size: Optional[int] = None,
        rrf_rank_constant: Optional[int] = None,
        language: Language = "zh-tw",
    ) -> SearchResponse:
        """
        Hybrid search combining BM25 and kNN with RRF fusion.

        RRF (Reciprocal Rank Fusion) combines rankings from different
        retrieval methods without requiring score normalization.

        RRF score = sum(1 / (rank_constant + rank_i))

        Note: This implementation computes RRF manually since ES RRF
        requires a paid license.

        Args:
            query: Search query text
            size: Number of results to return
            rrf_window_size: Number of results to consider for RRF (default: 100)
            rrf_rank_constant: RRF constant k (default: 60)
            language: Language of the query, passed through to search_knn() for
                correct embedding model selection.

        Returns:
            SearchResponse with results
        """
        import time
        start_time = time.time()

        window_size = rrf_window_size or self.RRF_WINDOW_SIZE
        rank_constant = rrf_rank_constant or self.RRF_RANK_CONSTANT

        # Get results from both methods
        bm25_response = self.search_bm25(query, size=window_size, language=language)
        try:
            knn_response = self.search_knn(query, size=window_size, language=language)
        except Exception as e:
            # kNN can fail when the index has mixed embedding dimensions (e.g. cross-language alias).
            # Fall back to BM25-only results so hybrid search degrades gracefully.
            logger.warning(
                "search_knn_failed_in_hybrid_fallback_to_bm25",
                extra={"query": query, "reason": str(e)},
            )
            knn_response = SearchResponse(results=[], total=0, took_ms=0, mode=SearchMode.KNN)

        # Compute RRF scores
        rrf_scores = self._compute_rrf_scores(
            bm25_response.results,
            knn_response.results,
            rank_constant=rank_constant,
        )

        # Build result map for deduplication
        result_map: Dict[str, SearchResult] = {}
        for result in bm25_response.results + knn_response.results:
            if result.episode_id not in result_map:
                result_map[result.episode_id] = result

        # Sort by RRF score and take top results, capped per show for diversity
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        results = []
        show_counts: Dict[str, int] = {}
        for episode_id in sorted_ids:
            if len(results) >= size:
                break
            result = result_map[episode_id]
            show_id = result.show_id or "__unknown__"
            if show_counts.get(show_id, 0) >= self.MAX_PER_SHOW:
                continue
            show_counts[show_id] = show_counts.get(show_id, 0) + 1
            results.append(SearchResult(
                episode_id=result.episode_id,
                title=result.title,
                description=result.description,
                score=rrf_scores[episode_id],
                show_title=result.show_title,
                show_id=result.show_id,
                published_at=result.published_at,
                duration_sec=result.duration_sec,
            ))

        took_ms = int((time.time() - start_time) * 1000)
        total = len(rrf_scores)

        logger.info(
            "search_hybrid",
            extra={
                "query": query,
                "results": len(results),
                "total": total,
                "took_ms": took_ms,
                "rrf_window_size": window_size,
                "rrf_rank_constant": rank_constant,
                "bm25_results": len(bm25_response.results),
                "knn_results": len(knn_response.results),
            },
        )

        return SearchResponse(
            results=results,
            total=total,
            took_ms=took_ms,
            mode=SearchMode.HYBRID,
        )

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        size: int = 10,
        language: Language = "zh-tw",
        **kwargs,
    ) -> SearchResponse:
        """
        Unified search interface.

        Args:
            query: Search query text
            mode: Search mode (BM25, KNN, HYBRID, or EXACT)
            size: Number of results to return
            language: Language of the query, passed to KNN/hybrid embedding.
                BM25 and EXACT modes ignore this parameter.
            **kwargs: Additional arguments forwarded to the underlying search method

        Returns:
            SearchResponse with results
        """
        if mode == SearchMode.BM25:
            return self.search_bm25(query, size=size, language=language)
        elif mode == SearchMode.KNN:
            return self.search_knn(query, size=size, language=language)
        elif mode == SearchMode.EXACT:
            return self.search_exact(query, size=size, language=language)
        else:
            return self.search_hybrid(query, size=size, language=language, **kwargs)
