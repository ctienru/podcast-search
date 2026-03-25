"""
Tests for evaluate_language_detection.py

All tests are unit tests — no Elasticsearch required.
Tests cover: detect_language(), evaluate_routing(), gate_status().
"""

import pytest

from scripts.evaluate_language_detection import (
    detect_language,
    evaluate_routing,
    evaluate_content_detection,
    gate_status,
    THRESHOLDS,
)


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------

class TestDetectLanguage:
    def test_chinese_text_returns_zh(self):
        assert detect_language("台灣科技新聞播客") == "zh"

    def test_english_text_returns_en(self):
        assert detect_language("technology and business podcast") == "en"

    def test_mixed_mostly_chinese_returns_zh(self):
        # Chinese chars / total non-space chars > 0.30 threshold
        # "科技新聞播客" (6) vs "tech" (4) → 6/10 = 60%
        assert detect_language("科技新聞播客 tech") == "zh"

    def test_empty_string_returns_unknown(self):
        assert detect_language("") == "unknown"

    def test_url_only_returns_en(self):
        # URLs/emails have no Chinese chars → en
        assert detect_language("https://example.com/feed.xml") == "en"

    def test_simplified_chinese_returns_zh(self):
        assert detect_language("每周科技资讯与投资理财分析") == "zh"


# ---------------------------------------------------------------------------
# Helpers to build sample fixtures
# ---------------------------------------------------------------------------

def _make_sample(index: str, correct_language: str, show_title: str = "Show", episode_title: str = "Ep") -> dict:
    return {
        "index": index,
        "correct_language": correct_language,
        "show_title": show_title,
        "episode_title": episode_title,
        "rss_language": correct_language,
        "description_snippet": "",
    }


def _make_samples_perfect(n_per_lang: int = 10) -> list:
    """All episodes routed to the correct index."""
    samples = []
    for _ in range(n_per_lang):
        samples.append(_make_sample("zh-tw", "zh-tw"))
        samples.append(_make_sample("zh-cn", "zh-cn"))
        samples.append(_make_sample("en", "en"))
    return samples


# ---------------------------------------------------------------------------
# evaluate_routing
# ---------------------------------------------------------------------------

class TestEvaluateRouting:
    def test_all_correct_gives_perfect_scores(self):
        samples = _make_samples_perfect(10)
        result = evaluate_routing(samples)
        assert result["accuracy"] == 1.0
        assert result["zh_confusion_rate"] == 0.0
        for lang in ("zh-tw", "zh-cn", "en"):
            assert result["per_class"][lang]["precision"] == 1.0
            assert result["per_class"][lang]["recall"] == 1.0

    def test_wrong_rss_tag_lowers_en_precision(self):
        # 3 zh-tw episodes incorrectly in the en index
        samples = _make_samples_perfect(10)
        for i in range(3):
            samples.append(_make_sample("en", "zh-tw", show_title=f"Wrong Show {i}"))
        result = evaluate_routing(samples)
        en_precision = result["per_class"]["en"]["precision"]
        assert en_precision < 1.0
        assert len(result["errors"]) == 3

    def test_zh_tw_in_zh_cn_index_counts_as_zh_confusion(self):
        samples = _make_samples_perfect(5)
        # 1 zh-tw episode routed to zh-cn
        samples.append(_make_sample("zh-cn", "zh-tw", show_title="Confused Show"))
        result = evaluate_routing(samples)
        assert result["zh_confusion_rate"] > 0.0

    def test_zh_in_en_index_does_not_count_as_zh_confusion(self):
        # zh-tw in en index is a routing error but NOT a zh-tw/zh-cn confusion
        samples = _make_samples_perfect(5)
        samples.append(_make_sample("en", "zh-tw"))
        result = evaluate_routing(samples)
        # zh confusion only counts zh-tw↔zh-cn swaps
        zh_samples = [s for s in samples if s["correct_language"] in ("zh-tw", "zh-cn")]
        confused = sum(
            1 for s in zh_samples
            if s["index"] in ("zh-tw", "zh-cn") and s["index"] != s["correct_language"]
        )
        assert confused == 0
        assert result["zh_confusion_rate"] == 0.0

    def test_errors_list_contains_mislabeled_episodes(self):
        samples = _make_samples_perfect(5)
        samples.append(_make_sample("en", "zh-tw", show_title="Bad Show", episode_title="Bad Ep"))
        result = evaluate_routing(samples)
        error_shows = [e["show"] for e in result["errors"]]
        assert "Bad Show" in error_shows


# ---------------------------------------------------------------------------
# gate_status
# ---------------------------------------------------------------------------

class TestGateStatus:
    def _routing_gate(self, precision_min: float, recall_min: float, zh_confusion: float) -> dict:
        return {
            "gate": {
                "precision_min": precision_min,
                "recall_min": recall_min,
                "zh_confusion_rate": zh_confusion,
            }
        }

    def _detection_gate(self) -> dict:
        return {"gate": {"precision_min": 1.0, "recall_min": 1.0}}

    def test_all_above_threshold_is_pass(self):
        routing = self._routing_gate(0.97, 0.95, 0.0)
        assert gate_status(routing, self._detection_gate()) == "PASS"

    def test_precision_below_threshold_is_fail(self):
        routing = self._routing_gate(0.93, 0.95, 0.0)  # 0.93 < 0.95
        assert gate_status(routing, self._detection_gate()) == "FAIL"

    def test_recall_below_threshold_is_fail(self):
        routing = self._routing_gate(0.97, 0.88, 0.0)  # 0.88 < 0.90
        assert gate_status(routing, self._detection_gate()) == "FAIL"

    def test_zh_confusion_above_threshold_is_fail(self):
        routing = self._routing_gate(0.97, 0.95, 0.06)  # 0.06 > 0.05
        assert gate_status(routing, self._detection_gate()) == "FAIL"

    def test_exactly_at_precision_threshold_is_pass(self):
        routing = self._routing_gate(THRESHOLDS["precision"], THRESHOLDS["recall"], 0.0)
        assert gate_status(routing, self._detection_gate()) == "PASS"

    def test_exactly_at_confusion_threshold_is_pass(self):
        routing = self._routing_gate(0.97, 0.95, THRESHOLDS["zh_confusion_rate"])
        assert gate_status(routing, self._detection_gate()) == "PASS"
