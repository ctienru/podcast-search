"""Tests for EmbeddingInputBuilder."""

import pytest
from src.embedding.input_builder import (
    EmbeddingInputBuilder,
    EmbeddingInputConfig,
    EmbeddingInput,
)


@pytest.fixture
def default_builder():
    """Create an EmbeddingInputBuilder with default config."""
    return EmbeddingInputBuilder()


@pytest.fixture
def custom_builder():
    """Create an EmbeddingInputBuilder with custom config."""
    config = EmbeddingInputConfig(
        title_weight=2,
        max_tokens=128,
        chars_per_token=3.0,
        include_show_title=True,
    )
    return EmbeddingInputBuilder(config)


@pytest.fixture
def sample_cleaned_episode():
    """Sample Layer 2 cleaned episode data."""
    return {
        "episode_id": "episode:apple:123:abc123",
        "show_id": "show:apple:123",
        "cleaned": {
            "normalized": {
                "title": "Test Episode Title",
                "description": "This is the cleaned description of the episode.",
            },
        },
    }


class TestEmbeddingInputConfig:
    """Test EmbeddingInputConfig defaults."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingInputConfig()

        assert config.include_show_title is True
        assert config.title_weight == 3
        assert config.max_tokens == 256
        assert config.chars_per_token == 2.5
        assert config.model_family == "sentence-transformers"
        assert config.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert config.embedding_dim == 384
        assert config.normalize_embeddings is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmbeddingInputConfig(
            title_weight=5,
            max_tokens=512,
            include_show_title=False,
        )

        assert config.title_weight == 5
        assert config.max_tokens == 512
        assert config.include_show_title is False


class TestBuild:
    """Test the build method."""

    def test_builds_embedding_input(self, default_builder, sample_cleaned_episode):
        """Test building embedding input from cleaned episode."""
        result = default_builder.build(sample_cleaned_episode)

        assert isinstance(result, EmbeddingInput)
        assert result.episode_id == "episode:apple:123:abc123"
        assert result.show_id == "show:apple:123"
        assert result.title == "Test Episode Title"

    def test_includes_show_title_when_provided(
        self, default_builder, sample_cleaned_episode
    ):
        """Test that show title is included when provided."""
        result = default_builder.build(sample_cleaned_episode, show_title="My Podcast")

        assert result.show_title == "My Podcast"
        assert "My Podcast:" in result.text

    def test_excludes_show_title_when_not_provided(
        self, default_builder, sample_cleaned_episode
    ):
        """Test that show title is excluded when not provided."""
        result = default_builder.build(sample_cleaned_episode)

        assert result.show_title is None
        assert not result.text.startswith(":")

    def test_excludes_show_title_when_config_disabled(self, sample_cleaned_episode):
        """Test that show title is excluded when config disables it."""
        config = EmbeddingInputConfig(include_show_title=False)
        builder = EmbeddingInputBuilder(config)

        result = builder.build(sample_cleaned_episode, show_title="My Podcast")

        # show_title should not appear in text even if provided
        assert "My Podcast:" not in result.text

    def test_applies_title_weight(self, sample_cleaned_episode):
        """Test that title is repeated according to weight."""
        config = EmbeddingInputConfig(title_weight=3)
        builder = EmbeddingInputBuilder(config)

        result = builder.build(sample_cleaned_episode)

        # Title should appear 3 times
        assert result.text.count("Test Episode Title") == 3
        assert result.title_weight == 3

    def test_includes_description(self, default_builder, sample_cleaned_episode):
        """Test that description is included in text."""
        result = default_builder.build(sample_cleaned_episode)

        assert "cleaned description" in result.text
        assert result.description_used == sample_cleaned_episode["cleaned"]["normalized"]["description"]


class TestTruncation:
    """Test text truncation behavior."""

    def test_truncates_long_description(self):
        """Test that long descriptions are truncated."""
        config = EmbeddingInputConfig(
            max_tokens=50,  # Very short limit
            chars_per_token=2.5,
            title_weight=1,
        )
        builder = EmbeddingInputBuilder(config)

        cleaned_episode = {
            "episode_id": "ep_trunc",
            "show_id": "show_trunc",
            "cleaned": {
                "normalized": {
                    "title": "Short",
                    "description": "A" * 500,  # Long description
                },
            },
        }

        result = builder.build(cleaned_episode)

        assert result.was_truncated is True
        assert len(result.description_used) < 500
        assert result.estimated_tokens <= config.max_tokens + 10  # Allow some margin

    def test_does_not_truncate_short_description(
        self, default_builder, sample_cleaned_episode
    ):
        """Test that short descriptions are not truncated."""
        result = default_builder.build(sample_cleaned_episode)

        assert result.was_truncated is False
        assert result.description_used == sample_cleaned_episode["cleaned"]["normalized"]["description"]

    def test_truncation_accounts_for_show_title(self):
        """Test that truncation accounts for show title length."""
        config = EmbeddingInputConfig(
            max_tokens=100,
            chars_per_token=2.5,
            title_weight=1,
        )
        builder = EmbeddingInputBuilder(config)

        cleaned_episode = {
            "episode_id": "ep_show",
            "show_id": "show_show",
            "cleaned": {
                "normalized": {
                    "title": "Title",
                    "description": "B" * 300,
                },
            },
        }

        # Without show title
        result_without = builder.build(cleaned_episode)

        # With show title
        result_with = builder.build(cleaned_episode, show_title="Very Long Podcast Name")

        # Result with show title should have shorter description
        assert len(result_with.description_used) <= len(result_without.description_used)


class TestEstimatedTokens:
    """Test token estimation."""

    def test_estimates_tokens(self, default_builder, sample_cleaned_episode):
        """Test that tokens are estimated."""
        result = default_builder.build(sample_cleaned_episode)

        assert result.estimated_tokens > 0
        assert result.max_tokens == 256  # Default

    def test_estimated_tokens_proportional_to_text_length(self):
        """Test that estimated tokens scale with text length."""
        config = EmbeddingInputConfig(chars_per_token=2.5)
        builder = EmbeddingInputBuilder(config)

        short_episode = {
            "episode_id": "ep_short",
            "show_id": "show_short",
            "cleaned": {
                "normalized": {
                    "title": "A",
                    "description": "Short",
                },
            },
        }

        long_episode = {
            "episode_id": "ep_long",
            "show_id": "show_long",
            "cleaned": {
                "normalized": {
                    "title": "Title",
                    "description": "C" * 100,
                },
            },
        }

        short_result = builder.build(short_episode)
        long_result = builder.build(long_episode)

        assert long_result.estimated_tokens > short_result.estimated_tokens


class TestModelConfig:
    """Test model configuration in output."""

    def test_includes_model_config(self, default_builder, sample_cleaned_episode):
        """Test that model configuration is included."""
        result = default_builder.build(sample_cleaned_episode)

        assert result.model_family == "sentence-transformers"
        assert result.model_name == "paraphrase-multilingual-MiniLM-L12-v2"
        assert result.normalize_embeddings is True

    def test_uses_custom_model_config(self, sample_cleaned_episode):
        """Test that custom model config is used."""
        config = EmbeddingInputConfig(
            model_family="custom-family",
            model_name="custom-model",
            normalize_embeddings=False,
        )
        builder = EmbeddingInputBuilder(config)

        result = builder.build(sample_cleaned_episode)

        assert result.model_family == "custom-family"
        assert result.model_name == "custom-model"
        assert result.normalize_embeddings is False


class TestToLayer3Dict:
    """Test Layer 3 dict conversion."""

    def test_converts_to_dict(self, default_builder, sample_cleaned_episode):
        """Test conversion to Layer 3 dict structure."""
        embedding_input = default_builder.build(sample_cleaned_episode)
        result = default_builder.to_layer3_dict(embedding_input)

        assert result["episode_id"] == "episode:apple:123:abc123"
        assert result["show_id"] == "show:apple:123"
        assert "embedding_input" in result

    def test_includes_text(self, default_builder, sample_cleaned_episode):
        """Test that text is included in dict."""
        embedding_input = default_builder.build(sample_cleaned_episode)
        result = default_builder.to_layer3_dict(embedding_input)

        assert "text" in result["embedding_input"]
        assert len(result["embedding_input"]["text"]) > 0

    def test_includes_components(self, default_builder, sample_cleaned_episode):
        """Test that components are included."""
        embedding_input = default_builder.build(
            sample_cleaned_episode, show_title="Test Podcast"
        )
        result = default_builder.to_layer3_dict(embedding_input)

        components = result["embedding_input"]["components"]
        assert "show_title" in components
        assert "title" in components
        assert "title_weight" in components
        assert "description_used" in components
        assert "max_tokens" in components
        assert "estimated_tokens" in components
        assert "was_truncated" in components

    def test_includes_model_config(self, default_builder, sample_cleaned_episode):
        """Test that model config is included."""
        embedding_input = default_builder.build(sample_cleaned_episode)
        result = default_builder.to_layer3_dict(embedding_input)

        model_config = result["embedding_input"]["model_config"]
        assert "model_family" in model_config
        assert "model_name" in model_config
        assert "normalize_embeddings" in model_config


class TestTextFormat:
    """Test text format combinations."""

    def test_format_without_show_title(self):
        """Test text format without show title."""
        config = EmbeddingInputConfig(title_weight=2)
        builder = EmbeddingInputBuilder(config)

        episode = {
            "episode_id": "ep_1",
            "show_id": "show_1",
            "cleaned": {
                "normalized": {
                    "title": "Episode",
                    "description": "Description",
                },
            },
        }

        result = builder.build(episode)

        # Format: "{title} {title} {description}"
        assert result.text == "Episode Episode Description"

    def test_format_with_show_title(self):
        """Test text format with show title."""
        config = EmbeddingInputConfig(title_weight=2)
        builder = EmbeddingInputBuilder(config)

        episode = {
            "episode_id": "ep_1",
            "show_id": "show_1",
            "cleaned": {
                "normalized": {
                    "title": "Episode",
                    "description": "Description",
                },
            },
        }

        result = builder.build(episode, show_title="Podcast")

        # Format: "{show_title}: {title} {title} {description}"
        assert result.text == "Podcast: Episode Episode Description"

    def test_format_without_description(self):
        """Test text format when description is empty."""
        config = EmbeddingInputConfig(title_weight=2)
        builder = EmbeddingInputBuilder(config)

        episode = {
            "episode_id": "ep_1",
            "show_id": "show_1",
            "cleaned": {
                "normalized": {
                    "title": "Episode",
                    "description": "",
                },
            },
        }

        result = builder.build(episode, show_title="Podcast")

        # Format: "{show_title}: {title} {title}" (no trailing space)
        assert result.text == "Podcast: Episode Episode"


class TestEdgeCases:
    """Test edge cases."""

    def test_handles_unicode_content(self, default_builder):
        """Test handling of Unicode content."""
        episode = {
            "episode_id": "ep_unicode",
            "show_id": "show_unicode",
            "cleaned": {
                "normalized": {
                    "title": "第一集：測試",
                    "description": "中文描述 with émojis 🎙️",
                },
            },
        }

        result = default_builder.build(episode, show_title="我的 Podcast")

        assert "第一集：測試" in result.text
        assert "中文描述" in result.text
        assert "我的 Podcast" in result.text

    def test_handles_empty_title(self, default_builder):
        """Test handling of empty title."""
        episode = {
            "episode_id": "ep_empty_title",
            "show_id": "show_empty",
            "cleaned": {
                "normalized": {
                    "title": "",
                    "description": "Has description only",
                },
            },
        }

        result = default_builder.build(episode)

        assert "Has description only" in result.text
        assert result.title == ""

    def test_handles_title_weight_zero(self):
        """Test handling of title weight zero."""
        config = EmbeddingInputConfig(title_weight=0)
        builder = EmbeddingInputBuilder(config)

        episode = {
            "episode_id": "ep_no_weight",
            "show_id": "show_no_weight",
            "cleaned": {
                "normalized": {
                    "title": "Title",
                    "description": "Description",
                },
            },
        }

        result = builder.build(episode)

        # Title should not be repeated but structure should work
        assert result.title_weight == 0

    def test_handles_very_long_title(self):
        """Test handling of very long title."""
        config = EmbeddingInputConfig(max_tokens=100, chars_per_token=2.5, title_weight=1)
        builder = EmbeddingInputBuilder(config)

        episode = {
            "episode_id": "ep_long_title",
            "show_id": "show_long",
            "cleaned": {
                "normalized": {
                    "title": "T" * 200,  # Very long title
                    "description": "Short",
                },
            },
        }

        result = builder.build(episode)

        # Should handle gracefully, possibly truncating description
        assert result.title == "T" * 200
        assert len(result.description_used) <= len("Short")
