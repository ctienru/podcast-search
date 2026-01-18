"""Tests for ElasticsearchService."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from elasticsearch.exceptions import RequestError, ConnectionError
from src.services.es_service import ElasticsearchService


@pytest.fixture
def mock_es_client():
    """Create a mock Elasticsearch client."""
    with patch('src.services.es_service.Elasticsearch') as mock_es_class:
        mock_client = MagicMock()
        mock_es_class.return_value = mock_client
        yield mock_client


class TestIndexExists:
    """Test the index_exists() method."""

    def test_index_exists_true(self, mock_es_client):
        """Test when index exists."""
        mock_es_client.indices.exists.return_value = True

        service = ElasticsearchService()
        result = service.index_exists("shows_v1")

        assert result is True
        mock_es_client.indices.exists.assert_called_once_with(index="shows_v1")

    def test_index_exists_false(self, mock_es_client):
        """Test when index doesn't exist."""
        mock_es_client.indices.exists.return_value = False

        service = ElasticsearchService()
        result = service.index_exists("nonexistent")

        assert result is False


class TestAliasExists:
    """Test the alias_exists() method."""

    def test_alias_exists_true(self, mock_es_client):
        """Test when alias exists."""
        mock_es_client.indices.exists_alias.return_value = True

        service = ElasticsearchService()
        result = service.alias_exists("shows")

        assert result is True
        mock_es_client.indices.exists_alias.assert_called_once_with(name="shows")

    def test_alias_exists_false(self, mock_es_client):
        """Test when alias doesn't exist."""
        mock_es_client.indices.exists_alias.return_value = False

        service = ElasticsearchService()
        result = service.alias_exists("nonexistent")

        assert result is False


class TestCreateIndex:
    """Test the create_index() method."""

    def test_create_index_success(self, mock_es_client):
        """Test successful index creation."""
        service = ElasticsearchService()

        mapping = {"mappings": {"properties": {}}}
        result = service.create_index("shows_v1", mapping)

        assert result is True
        mock_es_client.indices.create.assert_called_once_with(
            index="shows_v1",
            body=mapping
        )

    def test_create_index_already_exists(self, mock_es_client):
        """Test creating an index that already exists."""
        mock_es_client.indices.create.side_effect = RequestError(
            message="resource_already_exists_exception",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        result = service.create_index("shows_v1", {})

        assert result is False

    def test_create_index_connection_error(self, mock_es_client):
        """Test index creation with connection error."""
        mock_es_client.indices.create.side_effect = ConnectionError(
            message="Connection refused",
            errors=[]
        )

        service = ElasticsearchService()

        with pytest.raises(ConnectionError):
            service.create_index("shows_v1", {})

    def test_create_index_other_request_error(self, mock_es_client):
        """Test index creation with other request error."""
        mock_es_client.indices.create.side_effect = RequestError(
            message="Invalid mapping",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        with pytest.raises(RequestError):
            service.create_index("shows_v1", {})


class TestReindex:
    """Test the reindex() method."""

    def test_reindex_success(self, mock_es_client):
        """Test successful reindexing."""
        mock_es_client.reindex.return_value = {
            "total": 1000,
            "created": 1000
        }

        service = ElasticsearchService()

        result = service.reindex("shows_v1", "shows_v2")

        assert result is True
        mock_es_client.reindex.assert_called_once_with(
            body={
                "source": {"index": "shows_v1"},
                "dest": {"index": "shows_v2"}
            },
            wait_for_completion=True
        )

    def test_reindex_request_error(self, mock_es_client):
        """Test reindexing with request error."""
        mock_es_client.reindex.side_effect = RequestError(
            message="Source index not found",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        with pytest.raises(RequestError):
            service.reindex("nonexistent", "shows_v2")

    def test_reindex_connection_error(self, mock_es_client):
        """Test reindexing with connection error."""
        mock_es_client.reindex.side_effect = ConnectionError(
            message="Connection refused",
            errors=[]
        )

        service = ElasticsearchService()

        with pytest.raises(ConnectionError):
            service.reindex("shows_v1", "shows_v2")


class TestUpdateAliases:
    """Test the update_aliases() method."""

    def test_update_aliases_success(self, mock_es_client):
        """Test successful alias update."""
        service = ElasticsearchService()

        actions = [
            {"remove": {"alias": "shows", "index": "*"}},
            {"add": {"alias": "shows", "index": "shows_v2"}}
        ]

        result = service.update_aliases(actions)

        assert result is True
        mock_es_client.indices.update_aliases.assert_called_once_with(
            body={"actions": actions}
        )

    def test_update_aliases_request_error(self, mock_es_client):
        """Test alias update with request error."""
        mock_es_client.indices.update_aliases.side_effect = RequestError(
            message="Invalid alias action",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        actions = [{"add": {"alias": "shows", "index": "shows_v2"}}]

        with pytest.raises(RequestError):
            service.update_aliases(actions)

    def test_update_aliases_connection_error(self, mock_es_client):
        """Test alias update with connection error."""
        mock_es_client.indices.update_aliases.side_effect = ConnectionError(
            message="Connection refused",
            errors=[]
        )

        service = ElasticsearchService()

        actions = [{"add": {"alias": "shows", "index": "shows_v2"}}]

        with pytest.raises(ConnectionError):
            service.update_aliases(actions)


class TestDeleteIndex:
    """Test the delete_index() method."""

    def test_delete_index_success(self, mock_es_client):
        """Test successful index deletion."""
        service = ElasticsearchService()

        result = service.delete_index("shows_v1")

        assert result is True
        mock_es_client.indices.delete.assert_called_once_with(index="shows_v1")

    def test_delete_index_not_found(self, mock_es_client):
        """Test deleting an index that doesn't exist."""
        mock_es_client.indices.delete.side_effect = RequestError(
            message="index_not_found_exception",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        result = service.delete_index("nonexistent")

        assert result is False

    def test_delete_index_other_request_error(self, mock_es_client):
        """Test index deletion with other request error."""
        mock_es_client.indices.delete.side_effect = RequestError(
            message="Some other error",
            meta=Mock(),
            body={}
        )

        service = ElasticsearchService()

        with pytest.raises(RequestError):
            service.delete_index("shows_v1")

    def test_delete_index_connection_error(self, mock_es_client):
        """Test index deletion with connection error."""
        mock_es_client.indices.delete.side_effect = ConnectionError(
            message="Connection refused",
            errors=[]
        )

        service = ElasticsearchService()

        with pytest.raises(ConnectionError):
            service.delete_index("shows_v1")


class TestDocumentExists:
    """Test the document_exists() method."""

    def test_document_exists_true(self, mock_es_client):
        """Test when document exists."""
        mock_es_client.exists.return_value = True

        service = ElasticsearchService()
        result = service.document_exists("shows", "show_123")

        assert result is True
        mock_es_client.exists.assert_called_once_with(
            index="shows",
            id="show_123"
        )

    def test_document_exists_false(self, mock_es_client):
        """Test when document doesn't exist."""
        mock_es_client.exists.return_value = False

        service = ElasticsearchService()
        result = service.document_exists("shows", "nonexistent")

        assert result is False
