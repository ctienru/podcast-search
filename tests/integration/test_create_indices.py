"""Integration tests for CreateIndicesPipeline.

Requires a running Elasticsearch instance (ES_HOST env var).
Run with: pytest -m integration
Not included in the PR gate (unit tests only).
"""

import pytest
from elasticsearch import Elasticsearch

from src.config.settings import ES_HOST, INDEX_VERSION, MAPPINGS_DIR
from src.es.mapping_loader import MappingLoader
from src.pipelines.create_indices import CreateIndicesPipeline
from src.services.es_service import ElasticsearchService


_LANG_INDICES = [
    f"podcast-episodes-{lang}-v{INDEX_VERSION}"
    for lang in ("zh-tw", "zh-cn", "en")
]
_LANG_ALIASES = ["episodes-zh-tw", "episodes-zh-cn", "episodes-en"]


@pytest.fixture
def es_client() -> Elasticsearch:
    """Real ES client. Requires ES_HOST to be reachable."""
    return Elasticsearch(ES_HOST)


@pytest.fixture(autouse=True)
def cleanup(es_client: Elasticsearch):
    """Delete language-split test indices before and after each test."""
    def _delete_all() -> None:
        for index in _LANG_INDICES:
            es_client.indices.delete(index=index, ignore_unavailable=True)
        for alias in _LANG_ALIASES:
            if es_client.indices.exists_alias(name=alias):
                es_client.indices.delete_alias(index="*", name=alias)
            # Also delete any concrete index that shares the alias name
            es_client.indices.delete(index=alias, ignore_unavailable=True)

    _delete_all()
    yield
    _delete_all()


def _make_pipeline(es_client: Elasticsearch, *, enable_language_split: bool) -> CreateIndicesPipeline:
    return CreateIndicesPipeline(
        es_service=ElasticsearchService(client=es_client),
        mapping_loader=MappingLoader(MAPPINGS_DIR),
        enable_language_split=enable_language_split,
        reindex=False,
    )


@pytest.mark.integration
def test_creates_three_indices_when_language_split_enabled(es_client: Elasticsearch) -> None:
    """When enable_language_split=True, all three indices must exist after run()."""
    pipeline = _make_pipeline(es_client, enable_language_split=True)
    pipeline.run()

    for index in _LANG_INDICES:
        assert es_client.indices.exists(index=index), f"index missing: {index}"


@pytest.mark.integration
def test_each_alias_points_to_exactly_one_index(es_client: Elasticsearch) -> None:
    """Each language alias must resolve to exactly one backing index."""
    pipeline = _make_pipeline(es_client, enable_language_split=True)
    pipeline.run()

    for alias in _LANG_ALIASES:
        result = es_client.indices.get_alias(name=alias)
        assert len(result) == 1, f"alias {alias!r} should point to exactly one index, got {list(result)}"


@pytest.mark.integration
def test_preserves_v1_behaviour_when_flag_disabled(es_client: Elasticsearch) -> None:
    """When enable_language_split=False, no language-split indices should be created."""
    pipeline = _make_pipeline(es_client, enable_language_split=False)
    pipeline.run()

    for index in _LANG_INDICES:
        assert not es_client.indices.exists(index=index), f"unexpected index created: {index}"
