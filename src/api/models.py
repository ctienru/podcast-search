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
