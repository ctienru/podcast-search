"""
API Request/Response Models
"""

from pydantic import BaseModel, Field

from src.types import Language


class EmbedRequest(BaseModel):
    """Embedding request body."""
    texts: list[str] = Field(..., min_length=1, max_length=100, description="List of texts to embed (max 100)")
    language: Language = Field("en", description="Language of all texts: zh-tw, zh-cn, or en")


class EmbedResponse(BaseModel):
    """Embedding response body."""
    embeddings: list[list[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model name used for embedding")
    dimensions: int = Field(..., description="Embedding vector dimensions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    dimensions: int


# ── OpenAI-compatible /v1/embeddings ─────────────────────────────────────────

class OpenAIEmbedRequest(BaseModel):
    """OpenAI-compatible embedding request."""
    model: str = Field(..., description="Model identifier. Must be a supported model.")
    input: str | list[str] = Field(..., description="Text or list of texts to embed.")


class OpenAIEmbeddingObject(BaseModel):
    """Single embedding object in the response data array."""
    object: str = Field("embedding", description="Always 'embedding'.")
    index: int = Field(..., description="0-based position matching the input array.")
    embedding: list[float] = Field(..., description="Dense float vector.")


class OpenAIEmbedResponse(BaseModel):
    """OpenAI-compatible embedding response."""
    object: str = Field("list", description="Always 'list'.")
    data: list[OpenAIEmbeddingObject]
    model: str = Field(..., description="Model name echoed from the request.")
