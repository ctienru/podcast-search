"""
API Routes
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.models import EmbedRequest, EmbedResponse, HealthResponse
from src.embedding.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-loaded encoder (initialized on first request)
_encoder: EmbeddingEncoder | None = None


def get_encoder() -> EmbeddingEncoder:
    """Get or create the embedding encoder (singleton pattern)."""
    global _encoder
    if _encoder is None:
        logger.info("Initializing embedding encoder...")
        _encoder = EmbeddingEncoder()
        logger.info(f"Encoder ready: {_encoder.model_name}, dim={_encoder.embedding_dim}")
    return _encoder


@router.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint - also warms up the model."""
    encoder = get_encoder()
    return HealthResponse(
        status="ok",
        model=encoder.model_name,
        dimensions=encoder.embedding_dim,
    )


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Generate embeddings for a list of texts.

    - Max 100 texts per request
    - Returns normalized L2 embeddings
    """
    try:
        encoder = get_encoder()

        # Encode texts
        embeddings = encoder.encode_batch(req.texts, show_progress=False)

        # Convert to list of lists for JSON serialization
        embeddings_list = embeddings.tolist()

        return EmbedResponse(
            embeddings=embeddings_list,
            model=encoder.model_name,
            dimensions=encoder.embedding_dim,
        )
    except Exception as e:
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=str(e))
