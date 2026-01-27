"""
Extraneous Content Scorer

Calculate "extraneous content score" for paragraphs to evaluate cleaning effectiveness.
Difference from text_cleaner.py:
- text_cleaner: Removes extraneous content
- extraneous_scorer: Labels and scores (does not remove)

Usage:
    from src.evaluation.extraneous_scorer import ExtraneousScorer

    scorer = ExtraneousScorer()
    score = scorer.score_paragraph("This episode is sponsored by...")
    print(score.extraneous_score)  # 0.0 ~ 1.0
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParagraphScore:
    """Paragraph scoring result"""

    text: str
    features: Dict[str, float]
    extraneous_score: float  # 0.0 ~ 1.0
    is_extraneous: bool  # score > threshold
    matched_patterns: List[str] = field(default_factory=list)


@dataclass
class DocumentScore:
    """Document overall score"""

    episode_id: str
    total_paragraphs: int
    extraneous_paragraphs: int
    avg_extraneous_score: float
    max_extraneous_score: float
    paragraph_scores: List[ParagraphScore]


class ExtraneousScorer:
    """
    Paragraph-level extraneous content scorer

    Features:
    - sponsor_hit: Sponsor/advertisement keywords
    - promo_code_hit: Promo code/discount code
    - cta_hit: Subscribe/follow call-to-action
    - url_density: URL density
    - boilerplate_similarity: Similarity to known boilerplate
    """

    # ========== Chinese Patterns (from text_cleaner.py) ==========
    ZH_SPONSOR_PATTERNS = [
        r"(本集|本節目|本期|這集).*?(贊助|支持|呈現|提供)",
        r"(折扣碼|優惠碼|promo\s*code)",
        r"輸入.*?(折扣|優惠|code)",
        r"感謝.*?贊助",
    ]

    ZH_CTA_PATTERNS = [
        r"(訂閱|追蹤|關注|按讚).*(頻道|節目|podcast|youtube|ig|fb)",
        r"(留言|評論|review).*(apple|spotify|itunes|kkbox)",
        r"(每週|每月|weekly|monthly).*(更新|上架)",
        r"(歡迎|記得|別忘了).*(分享|轉發|訂閱)",
        r"(加入|追蹤|follow).*(@|ig|instagram|facebook|fb)",
        r"訂閱方案",
        r"更新頻率",
        r"訂閱平台",
        r"合作聯繫",
    ]

    ZH_HOSTING_PATTERNS = [
        r"Hosting\s+provided\s+by",
        r"Powered\s+by\s+(SoundOn|Firstory|KKBOX)",
    ]

    # ========== English Patterns ==========
    EN_SPONSOR_PATTERNS = [
        r"(sponsored|brought to you|powered)\s+by",
        r"use\s+code\s+.{1,20}\s+(for|to\s+get)",
        r"(discount|promo)\s+code",
        r"\d+%\s+off",
        r"Sponsors?:",
        r"special\s+offer",
        r"exclusive\s+deal",
    ]

    EN_CTA_PATTERNS = [
        r"(subscribe|follow|like)\s+(to\s+)?(the\s+)?(channel|podcast|show|newsletter)",
        r"(leave\s+a\s+)?(review|rating)\s+(on\s+)?(apple|spotify|itunes)",
        r"sign\s+up\s+(for|at)",
        r"join\s+(our|the)\s+(newsletter|community|circle)",
        r"follow\s+(us\s+)?(on\s+)?(instagram|twitter|x|facebook)",
        r"get\s+email\s+updates",
        r"Learn\s+more\s+about\s+sponsor\s+message\s+choices",
        r"Privacy\s+Policy",
    ]

    EN_PROMO_CODE_PATTERNS = [
        r"promo\s*code[:\s]+\w+",
        r"discount\s*code[:\s]+\w+",
        r"coupon\s*code[:\s]+\w+",
        r"use\s+code\s+[\'\"]?\w+[\'\"]?",
    ]

    # ========== General Patterns ==========
    URL_PATTERN = r"https?://\S+"
    EMAIL_PATTERN = r"[\w.-]+@[\w.-]+\.\w+"

    # Feature weights (adjustable)
    DEFAULT_WEIGHTS = {
        "sponsor_hit": 0.30,
        "promo_code_hit": 0.25,
        "cta_hit": 0.20,
        "url_density": 0.10,
        "hosting_hit": 0.15,
    }

    EXTRANEOUS_THRESHOLD = 0.3  # score > 0.3 is considered extraneous

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: float = 0.3,
        boilerplate_corpus: Optional[List[str]] = None,
    ):
        """
        Initialize scorer

        Args:
            weights: Feature weights, defaults to DEFAULT_WEIGHTS
            threshold: Extraneous determination threshold
            boilerplate_corpus: List of known boilerplate texts
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.threshold = threshold
        self.boilerplate_corpus = boilerplate_corpus or []

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        if not text:
            return "unknown"

        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
        total_chars = len(text.replace(" ", ""))

        if total_chars == 0:
            return "unknown"

        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"

    def _count_pattern_matches(
        self, text: str, patterns: List[str]
    ) -> tuple[int, List[str]]:
        """Count pattern matches and return matched patterns"""
        count = 0
        matched = []
        text_lower = text.lower()

        for pattern in patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                count += len(matches)
                matched.append(pattern[:40])  # Truncate long patterns

        return count, matched

    def _calculate_url_density(self, text: str) -> float:
        """Calculate URL density (URLs per 100 chars)"""
        if not text:
            return 0.0

        urls = re.findall(self.URL_PATTERN, text)
        url_count = len(urls)

        # Normalize to 0-1 range (assume >3 URLs per 500 chars is high)
        density = url_count / (len(text) / 500 + 1)
        return min(density / 3, 1.0)

    def _calculate_boilerplate_similarity(self, text: str) -> float:
        """Calculate similarity to boilerplate corpus"""
        if not self.boilerplate_corpus or not text:
            return 0.0

        text_words = set(text.lower().split())
        max_sim = 0.0

        for bp in self.boilerplate_corpus:
            bp_words = set(bp.lower().split())
            if text_words and bp_words:
                intersection = len(text_words & bp_words)
                union = len(text_words | bp_words)
                sim = intersection / union if union > 0 else 0
                max_sim = max(max_sim, sim)

        return max_sim

    def score_paragraph(self, text: str) -> ParagraphScore:
        """
        Calculate extraneous score for a single paragraph

        Args:
            text: Paragraph text

        Returns:
            ParagraphScore with features and score
        """
        if not text or not text.strip():
            return ParagraphScore(
                text=text,
                features={k: 0.0 for k in self.weights},
                extraneous_score=0.0,
                is_extraneous=False,
            )

        language = self._detect_language(text)
        all_matched_patterns = []

        # Select patterns based on language
        if language == "zh":
            sponsor_patterns = self.ZH_SPONSOR_PATTERNS + self.EN_SPONSOR_PATTERNS
            cta_patterns = self.ZH_CTA_PATTERNS + self.EN_CTA_PATTERNS
            promo_patterns = self.EN_PROMO_CODE_PATTERNS
            hosting_patterns = self.ZH_HOSTING_PATTERNS
        else:
            sponsor_patterns = self.EN_SPONSOR_PATTERNS
            cta_patterns = self.EN_CTA_PATTERNS
            promo_patterns = self.EN_PROMO_CODE_PATTERNS
            hosting_patterns = self.ZH_HOSTING_PATTERNS

        # Calculate features
        sponsor_count, sponsor_matched = self._count_pattern_matches(
            text, sponsor_patterns
        )
        cta_count, cta_matched = self._count_pattern_matches(text, cta_patterns)
        promo_count, promo_matched = self._count_pattern_matches(text, promo_patterns)
        hosting_count, hosting_matched = self._count_pattern_matches(
            text, hosting_patterns
        )

        all_matched_patterns.extend(sponsor_matched)
        all_matched_patterns.extend(cta_matched)
        all_matched_patterns.extend(promo_matched)
        all_matched_patterns.extend(hosting_matched)

        features = {
            "sponsor_hit": min(sponsor_count, 3) / 3,  # Normalize to 0-1
            "promo_code_hit": min(promo_count, 2) / 2,
            "cta_hit": min(cta_count, 3) / 3,
            "url_density": self._calculate_url_density(text),
            "hosting_hit": min(hosting_count, 1),
        }

        # Add boilerplate similarity if corpus is available
        if self.boilerplate_corpus:
            features["boilerplate_similarity"] = self._calculate_boilerplate_similarity(
                text
            )
            if "boilerplate_similarity" not in self.weights:
                self.weights["boilerplate_similarity"] = 0.15

        # Calculate weighted score
        score = sum(
            features.get(k, 0) * self.weights.get(k, 0) for k in self.weights
        )
        score = min(score, 1.0)

        return ParagraphScore(
            text=text,
            features=features,
            extraneous_score=score,
            is_extraneous=score > self.threshold,
            matched_patterns=all_matched_patterns,
        )

    def score_document(
        self,
        episode_id: str,
        paragraphs: List[str],
    ) -> DocumentScore:
        """
        Calculate extraneous score for an entire document

        Args:
            episode_id: Episode ID
            paragraphs: List of paragraphs

        Returns:
            DocumentScore with overall and per-paragraph scores
        """
        paragraph_scores = [self.score_paragraph(p) for p in paragraphs]

        if not paragraph_scores:
            return DocumentScore(
                episode_id=episode_id,
                total_paragraphs=0,
                extraneous_paragraphs=0,
                avg_extraneous_score=0.0,
                max_extraneous_score=0.0,
                paragraph_scores=[],
            )

        extraneous_count = sum(1 for ps in paragraph_scores if ps.is_extraneous)
        scores = [ps.extraneous_score for ps in paragraph_scores]

        return DocumentScore(
            episode_id=episode_id,
            total_paragraphs=len(paragraphs),
            extraneous_paragraphs=extraneous_count,
            avg_extraneous_score=sum(scores) / len(scores),
            max_extraneous_score=max(scores),
            paragraph_scores=paragraph_scores,
        )

    def to_dict(self, doc_score: DocumentScore) -> dict:
        """Convert DocumentScore to dict for JSON serialization"""
        return {
            "episode_id": doc_score.episode_id,
            "total_paragraphs": doc_score.total_paragraphs,
            "extraneous_paragraphs": doc_score.extraneous_paragraphs,
            "avg_extraneous_score": round(doc_score.avg_extraneous_score, 4),
            "max_extraneous_score": round(doc_score.max_extraneous_score, 4),
            "paragraph_scores": [
                {
                    "text": ps.text[:200] + "..." if len(ps.text) > 200 else ps.text,
                    "features": {k: round(v, 4) for k, v in ps.features.items()},
                    "extraneous_score": round(ps.extraneous_score, 4),
                    "is_extraneous": ps.is_extraneous,
                    "matched_patterns": ps.matched_patterns,
                }
                for ps in doc_score.paragraph_scores
            ],
        }
