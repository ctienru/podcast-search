"""
Embedding Input Builder

Transform Layer 2 (CLEANED) to Layer 3 (EMBEDDING_INPUT)

Features:
1. Show title prefix (provides context)
2. Title weighting (repeat N times)
3. Length truncation (max_tokens)
4. Record model configuration
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmbeddingInputConfig:
    """Embedding input configuration"""

    # Show title settings
    include_show_title: bool = True  # Whether to add show title prefix

    # Title weighting
    title_weight: int = 3  # Number of times to repeat title

    # Length limits
    max_tokens: int = 256  # Maximum token count (estimated)
    chars_per_token: float = 2.5  # Estimated for mixed CJK/English text

    # Model configuration
    model_family: str = "sentence-transformers"
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    normalize_embeddings: bool = True


@dataclass
class EmbeddingInput:
    """Layer 3: EMBEDDING_INPUT structure"""

    episode_id: str
    show_id: str
    text: str  # Final text to feed to model

    # Component details (for debugging)
    show_title: Optional[str]  # Show title (if available)
    title: str
    title_weight: int
    description_used: str
    max_tokens: int
    estimated_tokens: int
    was_truncated: bool

    # Model configuration
    model_family: str
    model_name: str
    normalize_embeddings: bool


class EmbeddingInputBuilder:
    """
    Transform Layer 2 (CLEANED) to Layer 3 (EMBEDDING_INPUT)

    Usage:
        builder = EmbeddingInputBuilder()
        result = builder.build(cleaned_episode, show_title="Podcast Name")
        # result.text -> text to feed to embedding model

    Text format:
        With show_title: "{show_title}: {title} {title} {description}"
        Without:         "{title} {title} {description}"
    """

    def __init__(self, config: Optional[EmbeddingInputConfig] = None):
        self.config = config or EmbeddingInputConfig()

    def build(
        self,
        cleaned_episode: dict,
        show_title: Optional[str] = None,
    ) -> EmbeddingInput:
        """
        Build Layer 3 EMBEDDING_INPUT from Layer 2 CLEANED JSON

        Args:
            cleaned_episode: Layer 2 JSON (from clean_episodes.py output)
            show_title: Optional show title for context (from ES or cache)

        Returns:
            EmbeddingInput with final text for embedding model
        """
        episode_id = cleaned_episode["episode_id"]
        show_id = cleaned_episode["show_id"]

        # Get cleaned title and description
        normalized = cleaned_episode["cleaned"]["normalized"]
        title = normalized["title"]
        description = normalized["description"]

        # Step 1: Build prefix (show_title if available)
        prefix = ""
        if self.config.include_show_title and show_title:
            prefix = f"{show_title}: "

        # Step 2: Title weighting (repeat N times)
        weighted_title = " ".join([title] * self.config.title_weight)

        # Step 3: Calculate available description length
        prefix_chars = len(prefix)
        title_chars = len(weighted_title)
        max_chars = int(self.config.max_tokens * self.config.chars_per_token)
        available_chars = max(0, max_chars - prefix_chars - title_chars - 1)  # -1 for space

        # Step 4: Truncate description
        was_truncated = len(description) > available_chars
        description_used = description[:available_chars] if was_truncated else description

        # Step 5: Combine final text
        # Format: "{show_title}: {title} {title} {description}"
        if description_used:
            text = f"{prefix}{weighted_title} {description_used}"
        else:
            text = f"{prefix}{weighted_title}"

        # Estimate token count
        estimated_tokens = int(len(text) / self.config.chars_per_token)

        return EmbeddingInput(
            episode_id=episode_id,
            show_id=show_id,
            text=text,
            show_title=show_title,
            title=title,
            title_weight=self.config.title_weight,
            description_used=description_used,
            max_tokens=self.config.max_tokens,
            estimated_tokens=estimated_tokens,
            was_truncated=was_truncated,
            model_family=self.config.model_family,
            model_name=self.config.model_name,
            normalize_embeddings=self.config.normalize_embeddings,
        )

    def to_layer3_dict(self, embedding_input: EmbeddingInput) -> dict:
        """
        Convert EmbeddingInput to Layer 3 JSON structure
        """
        return {
            "episode_id": embedding_input.episode_id,
            "show_id": embedding_input.show_id,
            "embedding_input": {
                "text": embedding_input.text,
                "components": {
                    "show_title": embedding_input.show_title,
                    "title": embedding_input.title,
                    "title_weight": embedding_input.title_weight,
                    "description_used": embedding_input.description_used,
                    "max_tokens": embedding_input.max_tokens,
                    "estimated_tokens": embedding_input.estimated_tokens,
                    "was_truncated": embedding_input.was_truncated,
                },
                "model_config": {
                    "model_family": embedding_input.model_family,
                    "model_name": embedding_input.model_name,
                    "normalize_embeddings": embedding_input.normalize_embeddings,
                },
            },
        }
