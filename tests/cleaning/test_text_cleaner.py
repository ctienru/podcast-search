"""Tests for PodcastTextCleaner."""

import pytest
from src.cleaning.text_cleaner import (
    PodcastTextCleaner,
    ParagraphInfo,
    CleaningStats,
    CleanedEpisode,
)


@pytest.fixture
def cleaner():
    """Create a PodcastTextCleaner instance."""
    return PodcastTextCleaner()


class TestCleanHtml:
    """Test HTML to plain text conversion."""

    def test_removes_html_tags(self, cleaner):
        """Test that HTML tags are removed."""
        html = "<p>Hello <strong>World</strong></p>"
        result = cleaner.clean_html(html)
        assert "<p>" not in result
        assert "<strong>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_preserves_newlines_from_block_elements(self, cleaner):
        """Test that block elements create newlines."""
        html = "<p>First paragraph</p><p>Second paragraph</p>"
        result = cleaner.clean_html(html)
        assert "\n" in result

    def test_removes_script_and_style(self, cleaner):
        """Test that script and style elements are removed."""
        html = "<p>Content</p><script>alert('xss')</script><style>.x{}</style>"
        result = cleaner.clean_html(html)
        assert "alert" not in result
        assert ".x" not in result
        assert "Content" in result

    def test_decodes_html_entities(self, cleaner):
        """Test that HTML entities are decoded."""
        html = "&amp; &lt; &gt; &quot; &#39;"
        result = cleaner.clean_html(html)
        assert "&" in result
        assert "<" in result
        assert ">" in result

    def test_handles_empty_input(self, cleaner):
        """Test handling of empty input."""
        assert cleaner.clean_html("") == ""
        assert cleaner.clean_html(None) == ""

    def test_extracts_text_from_links(self, cleaner):
        """Test that link text is preserved."""
        html = "<a href='https://example.com'>Click here</a>"
        result = cleaner.clean_html(html)
        assert "Click here" in result
        assert "href" not in result


class TestNormalizeText:
    """Test text normalization."""

    def test_normalizes_unicode(self, cleaner):
        """Test Unicode NFKC normalization."""
        # Full-width characters
        text = "Ｈｅｌｌｏ　Ｗｏｒｌｄ"
        result = cleaner.normalize_text(text)
        assert "Hello" in result
        assert "World" in result

    def test_removes_control_characters(self, cleaner):
        """Test removal of control characters."""
        text = "Hello\x00\x01\x02World"
        result = cleaner.normalize_text(text)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_preserves_newlines(self, cleaner):
        """Test that newlines are preserved."""
        text = "Line 1\nLine 2"
        result = cleaner.normalize_text(text)
        assert "\n" in result

    def test_collapses_multiple_newlines(self, cleaner):
        """Test that multiple newlines become double newline."""
        text = "Para 1\n\n\n\n\nPara 2"
        result = cleaner.normalize_text(text)
        assert "\n\n\n" not in result
        assert "Para 1" in result
        assert "Para 2" in result

    def test_strips_lines(self, cleaner):
        """Test that each line is stripped."""
        text = "  Line 1  \n  Line 2  "
        result = cleaner.normalize_text(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_handles_empty_input(self, cleaner):
        """Test handling of empty input."""
        assert cleaner.normalize_text("") == ""
        assert cleaner.normalize_text(None) == ""


class TestSplitParagraphs:
    """Test paragraph splitting."""

    def test_splits_on_blank_lines(self, cleaner):
        """Test splitting on blank lines."""
        text = "Paragraph 1\n\nParagraph 2\n\nParagraph 3"
        result = cleaner.split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "Paragraph 1"
        assert result[1] == "Paragraph 2"
        assert result[2] == "Paragraph 3"

    def test_filters_empty_paragraphs(self, cleaner):
        """Test that empty paragraphs are filtered."""
        text = "Para 1\n\n\n\nPara 2\n\n   \n\nPara 3"
        result = cleaner.split_paragraphs(text)
        assert len(result) == 3

    def test_handles_empty_input(self, cleaner):
        """Test handling of empty input."""
        assert cleaner.split_paragraphs("") == []
        assert cleaner.split_paragraphs(None) == []

    def test_single_paragraph(self, cleaner):
        """Test single paragraph without blank lines."""
        text = "Single paragraph with multiple lines\nbut no blank lines"
        result = cleaner.split_paragraphs(text)
        assert len(result) == 1


class TestDetectLanguage:
    """Test language detection."""

    def test_detects_chinese(self, cleaner):
        """Test detection of Chinese text."""
        text = "這是一個中文測試"
        assert cleaner.detect_language(text) == "zh"

    def test_detects_english(self, cleaner):
        """Test detection of English text."""
        text = "This is an English test"
        assert cleaner.detect_language(text) == "en"

    def test_detects_mixed_chinese_dominant(self, cleaner):
        """Test mixed text with Chinese dominant."""
        # Need enough Chinese characters to exceed 30% threshold
        text = "這是中文測試內容，包含很多中文字符"
        result = cleaner.detect_language(text)
        # With >30% Chinese characters, should detect as Chinese
        assert result == "zh"

    def test_detects_mixed_english_dominant(self, cleaner):
        """Test mixed text with English dominant."""
        text = "This is mostly English 測試"
        result = cleaner.detect_language(text)
        assert result == "en"

    def test_handles_empty_input(self, cleaner):
        """Test handling of empty input."""
        assert cleaner.detect_language("") == "unknown"
        assert cleaner.detect_language(None) == "unknown"

    def test_handles_whitespace_only(self, cleaner):
        """Test handling of whitespace-only input."""
        assert cleaner.detect_language("   ") == "unknown"


class TestIsBoilerplateParagraph:
    """Test boilerplate detection."""

    def test_detects_chinese_sponsor(self, cleaner):
        """Test detection of Chinese sponsor text."""
        text = "本集節目由 ABC 公司贊助播出"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "zh")
        assert is_bp is True
        assert "sponsor" in flags

    def test_detects_english_sponsor(self, cleaner):
        """Test detection of English sponsor text."""
        text = "This episode is sponsored by XYZ Company"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is True
        assert "sponsor" in flags

    def test_detects_chinese_cta(self, cleaner):
        """Test detection of Chinese call-to-action."""
        text = "記得訂閱我們的頻道"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "zh")
        assert is_bp is True
        assert "cta" in flags

    def test_detects_english_cta(self, cleaner):
        """Test detection of English call-to-action."""
        # Use text matching EN_CTA_PATTERNS - "follow us on instagram"
        text = "If you enjoyed this episode, follow us on Instagram and Twitter for more updates and behind-the-scenes content."
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is True
        assert "cta" in flags

    def test_detects_promo_code(self, cleaner):
        """Test detection of promo code."""
        text = "Use code SAVE20 for 20% off your purchase"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is True
        assert "sponsor" in flags

    def test_detects_hosting_provider(self, cleaner):
        """Test detection of hosting provider text."""
        text = "Hosting provided by SoundOn"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "zh")
        assert is_bp is True
        assert "hosting" in flags

    def test_detects_url_heavy_paragraph(self, cleaner):
        """Test detection of URL-heavy paragraph."""
        text = "Follow us at https://example.com and https://another.com"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is True
        assert "url_heavy" in flags

    def test_detects_timestamp_heavy_paragraph(self, cleaner):
        """Test detection of timestamp-heavy paragraph."""
        text = "(00:00) (05:30) (10:00) (15:00) x"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is True
        assert "timestamps" in flags

    def test_detects_short_cta(self, cleaner):
        """Test detection of short CTA text."""
        text = "訂閱更多"
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "zh")
        assert is_bp is True
        assert "short_cta" in flags

    def test_content_paragraph_not_flagged(self, cleaner):
        """Test that content paragraphs are not flagged."""
        text = "Today we discuss the latest developments in artificial intelligence and machine learning technology."
        is_bp, flags, rules = cleaner.is_boilerplate_paragraph(text, "en")
        assert is_bp is False
        assert len(flags) == 0


class TestBuildFrequencyTable:
    """Test frequency table building."""

    def test_builds_frequency_table(self, cleaner):
        """Test building frequency table for a show."""
        episodes = [
            {"description": "Unique content for episode 1\n\nRepeated boilerplate text"},
            {"description": "Unique content for episode 2\n\nRepeated boilerplate text"},
            {"description": "Unique content for episode 3\n\nRepeated boilerplate text"},
            {"description": "Unique content for episode 4\n\nRepeated boilerplate text"},
        ]

        cleaner.build_frequency_table("show_123", episodes)

        assert "show_123" in cleaner.show_paragraph_freq
        assert cleaner.show_paragraph_freq["show_123"]["total_episodes"] == 4

    def test_uses_content_encoded_if_available(self, cleaner):
        """Test that content_encoded is used when available."""
        episodes = [
            {"content_encoded": "Content from content:encoded tag"},
            {"description": "Content from description tag"},
        ]

        cleaner.build_frequency_table("show_456", episodes)

        assert "show_456" in cleaner.show_paragraph_freq


class TestIsFrequentParagraph:
    """Test frequent paragraph detection."""

    def test_detects_frequent_paragraph(self, cleaner):
        """Test detection of frequently repeated paragraph."""
        episodes = [
            {"description": "Unique 1\n\nThis is our standard outro"},
            {"description": "Unique 2\n\nThis is our standard outro"},
            {"description": "Unique 3\n\nThis is our standard outro"},
            {"description": "Unique 4\n\nThis is our standard outro"},
        ]

        cleaner.build_frequency_table("show_freq", episodes)

        # Frequent paragraph should be detected
        assert cleaner.is_frequent_paragraph(
            "This is our standard outro", "show_freq"
        )

    def test_does_not_flag_unique_paragraph(self, cleaner):
        """Test that unique paragraphs are not flagged."""
        episodes = [
            {"description": "Episode 1 unique content"},
            {"description": "Episode 2 different content"},
            {"description": "Episode 3 another topic"},
            {"description": "Episode 4 something else"},
        ]

        cleaner.build_frequency_table("show_unique", episodes)

        assert not cleaner.is_frequent_paragraph(
            "Episode 1 unique content", "show_unique"
        )

    def test_returns_false_for_unknown_show(self, cleaner):
        """Test that unknown show returns False."""
        assert not cleaner.is_frequent_paragraph("Some text", "unknown_show")

    def test_needs_minimum_episodes(self, cleaner):
        """Test that minimum episode count is required."""
        episodes = [
            {"description": "Short show content"},
            {"description": "Short show content"},
        ]

        cleaner.build_frequency_table("show_short", episodes)

        # Should return False because less than 4 episodes
        assert not cleaner.is_frequent_paragraph(
            "Short show content", "show_short"
        )


class TestCleanEpisode:
    """Test full episode cleaning."""

    def test_cleans_episode_with_boilerplate(self, cleaner):
        """Test cleaning an episode with boilerplate."""
        # Use separate paragraphs with blank lines to ensure they're treated as distinct
        result = cleaner.clean_episode(
            episode_id="ep_123",
            show_id="show_123",
            title="Test Episode Title",
            description="Main content here discussing interesting topics.\n\nThis episode is sponsored by XYZ Company with promo code SAVE20.",
        )

        assert isinstance(result, CleanedEpisode)
        assert result.episode_id == "ep_123"
        assert result.show_id == "show_123"
        assert "Main content here" in result.normalized_description
        assert "sponsored" not in result.normalized_description.lower()

    def test_returns_cleaning_stats(self, cleaner):
        """Test that cleaning stats are returned."""
        result = cleaner.clean_episode(
            episode_id="ep_456",
            show_id="show_456",
            title="Episode Title",
            description="<p>Content paragraph.</p><p>Subscribe to our newsletter!</p>",
        )

        assert result.stats.total_paragraphs >= 1
        assert result.stats.removed_paragraphs >= 0

    def test_uses_content_encoded_over_description(self, cleaner):
        """Test that content_encoded is preferred over description."""
        result = cleaner.clean_episode(
            episode_id="ep_789",
            show_id="show_789",
            title="Title",
            description="Short description",
            content_encoded="<p>Rich content from content:encoded tag.</p>",
        )

        assert "Rich content" in result.normalized_description

    def test_normalizes_title(self, cleaner):
        """Test that title is normalized."""
        result = cleaner.clean_episode(
            episode_id="ep_title",
            show_id="show_title",
            title="<b>Bold Title</b>  with   spaces",
            description="Description text",
        )

        assert "Bold Title" in result.normalized_title
        assert "<b>" not in result.normalized_title

    def test_handles_empty_description(self, cleaner):
        """Test handling of empty description."""
        result = cleaner.clean_episode(
            episode_id="ep_empty",
            show_id="show_empty",
            title="Title Only",
            description="",
        )

        assert result.normalized_description == ""
        assert result.stats.total_paragraphs == 0


class TestToLayer2Dict:
    """Test Layer 2 dict conversion."""

    def test_converts_to_dict(self, cleaner):
        """Test conversion to Layer 2 dict structure."""
        cleaned = cleaner.clean_episode(
            episode_id="ep_dict",
            show_id="show_dict",
            title="Test Title",
            description="Test description content.",
        )

        result = cleaner.to_layer2_dict(cleaned)

        assert result["episode_id"] == "ep_dict"
        assert result["show_id"] == "show_dict"
        assert "cleaned" in result
        assert "normalized" in result["cleaned"]
        assert "paragraphs" in result["cleaned"]
        assert "stats" in result["cleaned"]
        assert "cleaning_meta" in result

    def test_includes_paragraph_details(self, cleaner):
        """Test that paragraph details are included."""
        cleaned = cleaner.clean_episode(
            episode_id="ep_para",
            show_id="show_para",
            title="Title",
            description="Para 1\n\nPara 2",
        )

        result = cleaner.to_layer2_dict(cleaned)

        paragraphs = result["cleaned"]["paragraphs"]
        assert len(paragraphs) >= 1

        # Check paragraph structure
        para = paragraphs[0]
        assert "index" in para
        assert "text" in para
        assert "char_count" in para
        assert "flags" in para
        assert "kept" in para
        assert "rules_hit" in para

    def test_includes_stats(self, cleaner):
        """Test that stats are included."""
        cleaned = cleaner.clean_episode(
            episode_id="ep_stats",
            show_id="show_stats",
            title="Title",
            description="Content here",
        )

        result = cleaner.to_layer2_dict(cleaned)

        stats = result["cleaned"]["stats"]
        assert "raw_char_count" in stats
        assert "kept_char_count" in stats
        assert "total_paragraphs" in stats
        assert "kept_paragraphs" in stats
        assert "removed_paragraphs" in stats
        assert "removal_breakdown" in stats


class TestEdgeCases:
    """Test edge cases."""

    def test_handles_unicode_content(self, cleaner):
        """Test handling of Unicode content."""
        result = cleaner.clean_episode(
            episode_id="ep_unicode",
            show_id="show_unicode",
            title="測試 🎙️ Podcast",
            description="<p>中英混合內容 Mixed content</p>",
        )

        assert "測試" in result.normalized_title
        assert "🎙️" in result.normalized_title
        assert "中英混合內容" in result.normalized_description

    def test_handles_very_long_content(self, cleaner):
        """Test handling of very long content."""
        long_paragraph = "A" * 10000
        result = cleaner.clean_episode(
            episode_id="ep_long",
            show_id="show_long",
            title="Long Episode",
            description=long_paragraph,
        )

        assert len(result.normalized_description) == 10000

    def test_handles_special_characters(self, cleaner):
        """Test handling of special characters."""
        result = cleaner.clean_episode(
            episode_id="ep_special",
            show_id="show_special",
            title="Episode #1: Q&A!",
            description="<p>Content with &amp; and &lt;tags&gt;</p>",
        )

        assert "#1" in result.normalized_title
        assert "Q&A" in result.normalized_title
        assert "&" in result.normalized_description

    def test_preserves_numbers_and_dates(self, cleaner):
        """Test that numbers and dates are preserved."""
        result = cleaner.clean_episode(
            episode_id="ep_numbers",
            show_id="show_numbers",
            title="Episode 123",
            description="Published on 2024-01-15. Revenue: $1,000,000.",
        )

        assert "123" in result.normalized_title
        assert "2024-01-15" in result.normalized_description
        assert "$1,000,000" in result.normalized_description
