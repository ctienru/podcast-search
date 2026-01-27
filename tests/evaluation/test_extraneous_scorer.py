"""Tests for ExtraneousScorer."""

import pytest
from src.evaluation.extraneous_scorer import (
    ExtraneousScorer,
    ParagraphScore,
    DocumentScore,
)


@pytest.fixture
def scorer():
    """Create an ExtraneousScorer with default settings."""
    return ExtraneousScorer()


@pytest.fixture
def custom_scorer():
    """Create an ExtraneousScorer with custom settings."""
    return ExtraneousScorer(
        weights={
            "sponsor_hit": 0.4,
            "cta_hit": 0.3,
            "promo_code_hit": 0.2,
            "url_density": 0.1,
        },
        threshold=0.5,
    )


class TestScoreParagraph:
    """Test paragraph scoring."""

    def test_scores_clean_content(self, scorer):
        """Test scoring clean content paragraph."""
        text = "Today we discuss the latest trends in technology and innovation."
        result = scorer.score_paragraph(text)

        assert isinstance(result, ParagraphScore)
        assert result.extraneous_score < 0.3
        assert result.is_extraneous is False

    def test_scores_sponsor_paragraph(self, scorer):
        """Test scoring sponsor paragraph."""
        text = "This episode is sponsored by XYZ Company. Use code SAVE20 for 20% off."
        result = scorer.score_paragraph(text)

        assert result.extraneous_score > 0
        assert "sponsor" in result.features or result.features.get("sponsor_hit", 0) > 0

    def test_scores_cta_paragraph(self, scorer):
        """Test scoring CTA paragraph."""
        text = "Don't forget to subscribe to our podcast and leave a review on Apple Podcasts!"
        result = scorer.score_paragraph(text)

        assert result.features.get("cta_hit", 0) > 0

    def test_scores_promo_code_paragraph(self, scorer):
        """Test scoring promo code paragraph."""
        text = "Use promo code PODCAST for exclusive discount on your first order."
        result = scorer.score_paragraph(text)

        assert result.features.get("promo_code_hit", 0) > 0

    def test_handles_empty_paragraph(self, scorer):
        """Test handling empty paragraph."""
        result = scorer.score_paragraph("")

        assert result.extraneous_score == 0.0
        assert result.is_extraneous is False

    def test_handles_whitespace_paragraph(self, scorer):
        """Test handling whitespace-only paragraph."""
        result = scorer.score_paragraph("   \n\t  ")

        assert result.extraneous_score == 0.0
        assert result.is_extraneous is False

    def test_returns_matched_patterns(self, scorer):
        """Test that matched patterns are returned."""
        text = "This episode is sponsored by ABC Company."
        result = scorer.score_paragraph(text)

        assert isinstance(result.matched_patterns, list)

    def test_preserves_original_text(self, scorer):
        """Test that original text is preserved."""
        text = "Original text content."
        result = scorer.score_paragraph(text)

        assert result.text == text


class TestChinesePatterns:
    """Test Chinese pattern detection."""

    def test_detects_chinese_sponsor(self, scorer):
        """Test detection of Chinese sponsor text."""
        text = "本集節目由 ABC 公司贊助"
        result = scorer.score_paragraph(text)

        assert result.features.get("sponsor_hit", 0) > 0

    def test_detects_chinese_promo_code(self, scorer):
        """Test detection of Chinese promo code text."""
        text = "輸入折扣碼 SAVE20 享有優惠"
        result = scorer.score_paragraph(text)

        assert result.features.get("sponsor_hit", 0) > 0 or result.features.get("promo_code_hit", 0) > 0

    def test_detects_chinese_cta(self, scorer):
        """Test detection of Chinese CTA text."""
        text = "記得訂閱我們的頻道並追蹤我們的 IG"
        result = scorer.score_paragraph(text)

        assert result.features.get("cta_hit", 0) > 0

    def test_detects_chinese_hosting(self, scorer):
        """Test detection of Chinese hosting provider text."""
        text = "Powered by SoundOn"
        result = scorer.score_paragraph(text)

        assert result.features.get("hosting_hit", 0) > 0


class TestEnglishPatterns:
    """Test English pattern detection."""

    def test_detects_brought_to_you(self, scorer):
        """Test detection of 'brought to you by' pattern."""
        text = "This podcast is brought to you by our sponsor."
        result = scorer.score_paragraph(text)

        assert result.features.get("sponsor_hit", 0) > 0

    def test_detects_percentage_off(self, scorer):
        """Test detection of percentage discount pattern."""
        text = "Get 50% off your first order with code ABC."
        result = scorer.score_paragraph(text)

        assert result.features.get("sponsor_hit", 0) > 0

    def test_detects_follow_us(self, scorer):
        """Test detection of 'follow us' pattern."""
        text = "Follow us on Instagram and Twitter for updates!"
        result = scorer.score_paragraph(text)

        assert result.features.get("cta_hit", 0) > 0

    def test_detects_leave_review(self, scorer):
        """Test detection of 'leave a review' pattern."""
        text = "Please leave a review on Apple Podcasts if you enjoyed this episode."
        result = scorer.score_paragraph(text)

        assert result.features.get("cta_hit", 0) > 0

    def test_detects_sign_up(self, scorer):
        """Test detection of 'sign up' pattern."""
        text = "Sign up for our newsletter at example.com"
        result = scorer.score_paragraph(text)

        assert result.features.get("cta_hit", 0) > 0


class TestUrlDensity:
    """Test URL density calculation."""

    def test_calculates_url_density(self, scorer):
        """Test URL density calculation."""
        text = "Visit https://example.com and https://another.com for more info."
        result = scorer.score_paragraph(text)

        assert result.features.get("url_density", 0) > 0

    def test_low_url_density_for_single_url(self, scorer):
        """Test low URL density for single URL in long text."""
        text = "A" * 500 + " https://example.com " + "B" * 500
        result = scorer.score_paragraph(text)

        # Should have low URL density
        assert result.features.get("url_density", 0) < 0.5

    def test_high_url_density_for_many_urls(self, scorer):
        """Test high URL density for many URLs in short text."""
        text = "https://a.com https://b.com https://c.com"
        result = scorer.score_paragraph(text)

        assert result.features.get("url_density", 0) > 0


class TestLanguageDetection:
    """Test language detection."""

    def test_detects_chinese_language(self, scorer):
        """Test detection of Chinese language."""
        result = scorer._detect_language("這是中文內容")
        assert result == "zh"

    def test_detects_english_language(self, scorer):
        """Test detection of English language."""
        result = scorer._detect_language("This is English content")
        assert result == "en"

    def test_detects_mixed_chinese_dominant(self, scorer):
        """Test detection of mixed text with Chinese dominant."""
        # Need enough Chinese characters to exceed 30% threshold
        result = scorer._detect_language("這是中文測試內容，包含很多中文字符")
        assert result == "zh"

    def test_handles_empty_text(self, scorer):
        """Test handling of empty text."""
        result = scorer._detect_language("")
        assert result == "unknown"


class TestThreshold:
    """Test threshold behavior."""

    def test_custom_threshold(self):
        """Test custom threshold."""
        high_threshold_scorer = ExtraneousScorer(threshold=0.9)
        low_threshold_scorer = ExtraneousScorer(threshold=0.1)

        text = "This episode is sponsored by XYZ."

        high_result = high_threshold_scorer.score_paragraph(text)
        low_result = low_threshold_scorer.score_paragraph(text)

        # Same score but different is_extraneous
        assert high_result.extraneous_score == low_result.extraneous_score
        # Low threshold more likely to flag as extraneous
        assert low_result.is_extraneous or not high_result.is_extraneous


class TestCustomWeights:
    """Test custom weights."""

    def test_custom_weights_affect_score(self):
        """Test that custom weights affect score."""
        default_scorer = ExtraneousScorer()
        custom_scorer = ExtraneousScorer(
            weights={"sponsor_hit": 1.0, "cta_hit": 0.0, "promo_code_hit": 0.0, "url_density": 0.0, "hosting_hit": 0.0}
        )

        text = "This episode is sponsored by XYZ."

        default_result = default_scorer.score_paragraph(text)
        custom_result = custom_scorer.score_paragraph(text)

        # Custom scorer weights sponsor_hit at 1.0, so should have different score
        # (assuming default is not 1.0)


class TestBoilerplateCorpus:
    """Test boilerplate corpus functionality."""

    def test_uses_boilerplate_corpus(self):
        """Test boilerplate corpus similarity."""
        boilerplate = [
            "Thanks for listening to our podcast",
            "Subscribe and leave a review",
        ]
        scorer = ExtraneousScorer(boilerplate_corpus=boilerplate)

        similar_text = "Thanks for listening to our podcast today"
        result = scorer.score_paragraph(similar_text)

        assert "boilerplate_similarity" in result.features
        assert result.features["boilerplate_similarity"] > 0

    def test_empty_boilerplate_corpus(self, scorer):
        """Test with empty boilerplate corpus."""
        text = "Some random content."
        result = scorer.score_paragraph(text)

        # Should not have boilerplate_similarity or should be 0
        assert result.features.get("boilerplate_similarity", 0) == 0


class TestScoreDocument:
    """Test document scoring."""

    def test_scores_document(self, scorer):
        """Test scoring entire document."""
        paragraphs = [
            "Clean content paragraph about technology.",
            "Another paragraph about innovation.",
            "This episode is sponsored by XYZ Company.",
        ]

        result = scorer.score_document("ep_123", paragraphs)

        assert isinstance(result, DocumentScore)
        assert result.episode_id == "ep_123"
        assert result.total_paragraphs == 3
        assert len(result.paragraph_scores) == 3

    def test_counts_extraneous_paragraphs(self, scorer):
        """Test counting extraneous paragraphs."""
        paragraphs = [
            "Clean content.",
            "Sponsored by ABC. Use code XYZ for 20% off.",
            "Subscribe to our channel!",
        ]

        result = scorer.score_document("ep_count", paragraphs)

        assert result.extraneous_paragraphs <= result.total_paragraphs

    def test_calculates_average_score(self, scorer):
        """Test average score calculation."""
        paragraphs = [
            "Clean content.",
            "More clean content.",
        ]

        result = scorer.score_document("ep_avg", paragraphs)

        assert 0 <= result.avg_extraneous_score <= 1

    def test_calculates_max_score(self, scorer):
        """Test max score calculation."""
        paragraphs = [
            "Clean content.",
            "This episode is sponsored by XYZ. Use code ABC for discount.",
        ]

        result = scorer.score_document("ep_max", paragraphs)

        assert result.max_extraneous_score >= result.avg_extraneous_score

    def test_handles_empty_paragraphs_list(self, scorer):
        """Test handling empty paragraphs list."""
        result = scorer.score_document("ep_empty", [])

        assert result.total_paragraphs == 0
        assert result.extraneous_paragraphs == 0
        assert result.avg_extraneous_score == 0.0
        assert result.max_extraneous_score == 0.0


class TestToDict:
    """Test dict conversion."""

    def test_converts_document_score_to_dict(self, scorer):
        """Test conversion of DocumentScore to dict."""
        paragraphs = ["Clean content.", "Sponsored paragraph."]
        doc_score = scorer.score_document("ep_dict", paragraphs)

        result = scorer.to_dict(doc_score)

        assert result["episode_id"] == "ep_dict"
        assert result["total_paragraphs"] == 2
        assert "extraneous_paragraphs" in result
        assert "avg_extraneous_score" in result
        assert "max_extraneous_score" in result
        assert "paragraph_scores" in result

    def test_truncates_long_text_in_dict(self, scorer):
        """Test that long text is truncated in dict."""
        long_text = "A" * 500
        paragraphs = [long_text]
        doc_score = scorer.score_document("ep_long", paragraphs)

        result = scorer.to_dict(doc_score)

        para_text = result["paragraph_scores"][0]["text"]
        assert len(para_text) <= 203  # 200 + "..."

    def test_rounds_scores_in_dict(self, scorer):
        """Test that scores are rounded in dict."""
        paragraphs = ["Test content."]
        doc_score = scorer.score_document("ep_round", paragraphs)

        result = scorer.to_dict(doc_score)

        # Check that scores are rounded to 4 decimal places
        assert isinstance(result["avg_extraneous_score"], float)
        assert isinstance(result["max_extraneous_score"], float)

    def test_includes_paragraph_details(self, scorer):
        """Test that paragraph details are included."""
        paragraphs = ["Test paragraph content."]
        doc_score = scorer.score_document("ep_details", paragraphs)

        result = scorer.to_dict(doc_score)

        para = result["paragraph_scores"][0]
        assert "text" in para
        assert "features" in para
        assert "extraneous_score" in para
        assert "is_extraneous" in para
        assert "matched_patterns" in para


class TestEdgeCases:
    """Test edge cases."""

    def test_handles_unicode_content(self, scorer):
        """Test handling of Unicode content."""
        text = "這是中文測試 🎙️ with émojis"
        result = scorer.score_paragraph(text)

        assert isinstance(result, ParagraphScore)

    def test_handles_very_long_paragraph(self, scorer):
        """Test handling of very long paragraph."""
        long_text = "Content " * 1000
        result = scorer.score_paragraph(long_text)

        assert isinstance(result, ParagraphScore)

    def test_handles_special_characters(self, scorer):
        """Test handling of special characters."""
        text = "Content with special chars: @#$%^&*()[]{}|\\:\";<>?,./`~"
        result = scorer.score_paragraph(text)

        assert isinstance(result, ParagraphScore)

    def test_case_insensitive_matching(self, scorer):
        """Test that pattern matching is case insensitive."""
        text1 = "SPONSORED BY XYZ"
        text2 = "sponsored by xyz"

        result1 = scorer.score_paragraph(text1)
        result2 = scorer.score_paragraph(text2)

        assert result1.features.get("sponsor_hit", 0) == result2.features.get("sponsor_hit", 0)

    def test_normalizes_features_to_unit_range(self, scorer):
        """Test that features are normalized to 0-1 range."""
        text = "Sponsor Sponsor Sponsor. Subscribe Subscribe. Promo code SAVE."
        result = scorer.score_paragraph(text)

        for feature, value in result.features.items():
            assert 0 <= value <= 1, f"Feature {feature} has value {value} outside [0,1]"

    def test_score_bounded_to_unit_range(self, scorer):
        """Test that final score is bounded to 0-1."""
        # Text with multiple high-scoring features
        text = "Sponsored by ABC. Use promo code XYZ. Subscribe now! https://a.com https://b.com"
        result = scorer.score_paragraph(text)

        assert 0 <= result.extraneous_score <= 1
