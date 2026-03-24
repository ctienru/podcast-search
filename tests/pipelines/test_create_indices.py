"""Tests for CreateIndicesPipeline."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.pipelines.create_indices import CreateIndicesPipeline


class TestVersionedIndex:
    """Test the _versioned_index() helper method."""

    def test_versioned_index_naming(self):
        """Test that versioned index names are generated correctly."""
        pipeline = CreateIndicesPipeline(
            es_service=MagicMock(),
            mapping_loader=MagicMock(),
            index_version=5
        )

        assert pipeline._versioned_index("shows") == "shows_v5"
        assert pipeline._versioned_index("episodes") == "episodes_v5"

    def test_versioned_index_version_1(self):
        """Test versioned index naming for version 1."""
        pipeline = CreateIndicesPipeline(
            es_service=MagicMock(),
            mapping_loader=MagicMock(),
            index_version=1
        )

        assert pipeline._versioned_index("shows") == "shows_v1"


class TestEnsureAliasNameIsFree:
    """Test the ensure_alias_name_is_free() method."""

    def test_alias_already_exists_is_ok(self):
        """Test that existing alias is allowed (will be handled in switch_alias)."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = True
        mock_es.index_exists.return_value = False

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock()
        )

        # Should not raise
        pipeline.ensure_alias_name_is_free("shows")

    def test_concrete_index_exists_without_permission(self):
        """Test that concrete index with same name raises error without permission."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = False
        mock_es.index_exists.return_value = True

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock(),
            allow_delete_base_index=False
        )

        with pytest.raises(RuntimeError, match="alias_name_conflict"):
            pipeline.ensure_alias_name_is_free("shows")

    def test_concrete_index_deleted_with_permission(self):
        """Test that concrete index is deleted when permission is granted."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = False
        mock_es.index_exists.return_value = True

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock(),
            allow_delete_base_index=True
        )

        # Should not raise, should delete index
        pipeline.ensure_alias_name_is_free("shows")

        mock_es.delete_index.assert_called_once_with("shows")

    def test_no_conflict_scenario(self):
        """Test scenario where there's no conflict (neither index nor alias exists)."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = False
        mock_es.index_exists.return_value = False

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock()
        )

        # Should not raise or call any delete methods
        pipeline.ensure_alias_name_is_free("shows")

        mock_es.delete_index.assert_not_called()


class TestCreateVersionedIndex:
    """Test the create_versioned_index() method."""

    def test_creates_new_index(self):
        """Test that new index is created with correct mapping."""
        mock_es = MagicMock()
        mock_es.index_exists.return_value = False

        mock_loader = MagicMock()
        mock_mapping = {"mappings": {"properties": {}}}
        mock_loader.load.return_value = mock_mapping

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=3
        )

        result = pipeline.create_versioned_index("shows")

        assert result == "shows_v3"
        mock_loader.load.assert_called_once_with("shows")
        mock_es.create_index.assert_called_once_with("shows_v3", mock_mapping)

    def test_skips_existing_index(self):
        """Test that existing index is not recreated."""
        mock_es = MagicMock()
        mock_es.index_exists.return_value = True

        mock_loader = MagicMock()

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=3
        )

        result = pipeline.create_versioned_index("shows")

        assert result == "shows_v3"
        mock_loader.load.assert_not_called()
        mock_es.create_index.assert_not_called()


class TestReindexIfNeeded:
    """Test the reindex_if_needed() method."""

    def test_reindex_when_enabled_and_source_exists(self):
        """Test reindex when enabled and source exists."""
        mock_es = MagicMock()
        mock_es.index_exists.return_value = True

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock(),
            reindex=True
        )

        pipeline.reindex_if_needed("shows_v2", "shows_v3")

        mock_es.reindex.assert_called_once_with("shows_v2", "shows_v3")

    def test_skips_when_reindex_disabled(self):
        """Test that reindex is skipped when disabled."""
        mock_es = MagicMock()

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock(),
            reindex=False
        )

        pipeline.reindex_if_needed("shows_v2", "shows_v3")

        mock_es.reindex.assert_not_called()

    def test_skips_when_source_missing(self):
        """Test that reindex is skipped when source doesn't exist."""
        mock_es = MagicMock()
        mock_es.index_exists.return_value = False

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock(),
            reindex=True
        )

        pipeline.reindex_if_needed("shows_v2", "shows_v3")

        mock_es.reindex.assert_not_called()


class TestSwitchAlias:
    """Test the switch_alias() method."""

    def test_creates_new_alias(self):
        """Test creating a new alias when it doesn't exist."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = False
        mock_es.index_exists.return_value = False

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock()
        )

        pipeline.switch_alias("shows", "shows_v3")

        # Should only add the alias
        expected_actions = [
            {"add": {"alias": "shows", "index": "shows_v3"}}
        ]
        mock_es.update_aliases.assert_called_once_with(expected_actions)

    def test_switches_existing_alias(self):
        """Test switching an existing alias to a new index."""
        mock_es = MagicMock()
        mock_es.alias_exists.return_value = True
        mock_es.index_exists.return_value = False

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=MagicMock()
        )

        pipeline.switch_alias("shows", "shows_v3")

        # Should remove old alias and add new one
        expected_actions = [
            {"remove": {"alias": "shows", "index": "*"}},
            {"add": {"alias": "shows", "index": "shows_v3"}}
        ]
        mock_es.update_aliases.assert_called_once_with(expected_actions)


class TestRunForIndex:
    """Test the run_for_index() orchestration method."""

    def test_version_1_no_reindex(self):
        """Test version 1 creates index without reindexing."""
        mock_es = MagicMock()
        mock_es.index_exists.return_value = False
        mock_es.alias_exists.return_value = False

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"mappings": {}}

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=1,
            reindex=True  # Even with reindex enabled
        )

        pipeline.run_for_index("shows")

        # Should create index
        mock_es.create_index.assert_called_once_with("shows_v1", {"mappings": {}})

        # Should not attempt reindex (no old_index for version 1)
        mock_es.reindex.assert_not_called()

        # Should switch alias
        mock_es.update_aliases.assert_called_once()

    def test_version_upgrade_with_reindex(self):
        """Test upgrading from v4 to v5 with reindex."""
        mock_es = MagicMock()

        def index_exists_side_effect(index_name):
            if index_name == "shows_v4":
                return True  # Old version exists
            if index_name == "shows_v5":
                return False  # New version doesn't exist yet
            return False

        mock_es.index_exists.side_effect = index_exists_side_effect
        mock_es.alias_exists.return_value = False

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"mappings": {}}

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=5,
            reindex=True
        )

        pipeline.run_for_index("shows")

        # Should create new index
        mock_es.create_index.assert_called_once_with("shows_v5", {"mappings": {}})

        # Should reindex from old version
        mock_es.reindex.assert_called_once_with("shows_v4", "shows_v5")

        # Should switch alias
        mock_es.update_aliases.assert_called_once()

    def test_version_upgrade_fallback_to_alias(self):
        """Test fallback to alias when old versioned index doesn't exist."""
        mock_es = MagicMock()

        def index_exists_side_effect(index_name):
            if index_name == "shows_v4":
                return False  # Old versioned index doesn't exist
            if index_name == "shows_v5":
                return False  # New version doesn't exist yet
            if index_name == "shows":
                return True  # Alias points to a real index, so ES treats it as existing
            return False

        mock_es.index_exists.side_effect = index_exists_side_effect
        mock_es.alias_exists.side_effect = lambda alias: alias == "shows"  # Alias exists

        mock_loader = MagicMock()
        mock_loader.load.return_value = {"mappings": {}}

        pipeline = CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=5,
            reindex=True
        )

        pipeline.run_for_index("shows")

        # Should reindex from alias instead
        mock_es.reindex.assert_called_once_with("shows", "shows_v5")


class TestRunLanguageSplit:
    """Test the _run_language_split() orchestration method."""

    def _make_pipeline(self, index_version=1, reindex=False):
        mock_es = MagicMock()
        mock_es.index_exists.return_value = False
        mock_es.alias_exists.return_value = False
        mock_loader = MagicMock()
        mock_loader.load.return_value = {"mappings": {}}
        return CreateIndicesPipeline(
            es_service=mock_es,
            mapping_loader=mock_loader,
            index_version=index_version,
            reindex=reindex,
            enable_language_split=True,
        ), mock_es, mock_loader

    def test_shows_index_uses_podcast_prefix(self):
        """_run_language_split() must create podcast-shows_v1, not shows_v1."""
        pipeline, mock_es, mock_loader = self._make_pipeline()

        pipeline._run_language_split()

        created_indices = [call.args[0] for call in mock_es.create_index.call_args_list]
        assert "podcast-shows_v1" in created_indices
        assert "shows_v1" not in created_indices

    def test_shows_alias_is_shows(self):
        """shows alias must point to podcast-shows_v1."""
        pipeline, mock_es, _ = self._make_pipeline()

        pipeline._run_language_split()

        all_actions = []
        for call in mock_es.update_aliases.call_args_list:
            all_actions.extend(call.args[0])

        add_actions = [a["add"] for a in all_actions if "add" in a]
        shows_adds = [a for a in add_actions if a["alias"] == "shows"]
        assert len(shows_adds) == 1
        assert shows_adds[0]["index"] == "podcast-shows_v1"

    def test_shows_mapping_key_is_shows(self):
        """create_versioned_index for shows must use mapping_key='shows'."""
        pipeline, _, mock_loader = self._make_pipeline()

        pipeline._run_language_split()

        load_calls = [call.args[0] for call in mock_loader.load.call_args_list]
        assert "shows" in load_calls
        assert "podcast-shows" not in load_calls

    def test_episode_indices_created_for_all_languages(self):
        """_run_language_split() creates podcast-episodes-{lang}_v1 for zh-tw, zh-cn, en."""
        pipeline, mock_es, _ = self._make_pipeline()

        pipeline._run_language_split()

        created_indices = [call.args[0] for call in mock_es.create_index.call_args_list]
        assert "podcast-episodes-zh-tw_v1" in created_indices
        assert "podcast-episodes-zh-cn_v1" in created_indices
        assert "podcast-episodes-en_v1" in created_indices
