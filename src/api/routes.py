"""
API Routes
"""

import logging
from fastapi import APIRouter, HTTPException

from src.api.models import EmbedRequest, EmbedResponse, HealthResponse
from src.embedding.backend import EmbeddingBackend, LocalEmbeddingBackend
from src.types import Language

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-loaded backend (initialised on first request, shared for process lifetime)
_backend: EmbeddingBackend | None = None

# Model names reported in responses — kept in sync with backend._MODEL_MAP
_LANGUAGE_TO_MODEL: dict[str, str] = {
    "zh-tw": "BAAI/bge-base-zh-v1.5",
    "zh-cn": "BAAI/bge-base-zh-v1.5",
    "en":    "paraphrase-multilingual-MiniLM-L12-v2",
}
_LANGUAGE_TO_DIM: dict[str, int] = {
    "zh-tw": 768,
    "zh-cn": 768,
    "en":    384,
}


def get_backend() -> EmbeddingBackend:
    """Get or create the embedding backend (singleton per process)."""
    global _backend
    if _backend is None:
        logger.info("initialising_embedding_backend")
        _backend = LocalEmbeddingBackend()
        logger.info("embedding_backend_ready")
    return _backend


@router.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    get_backend()
    return HealthResponse(
        status="ok",
        model=_LANGUAGE_TO_MODEL["en"],
        dimensions=_LANGUAGE_TO_DIM["en"],
    )


@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """Generate embeddings for a list of texts.

    All texts are encoded with the model that corresponds to the requested
    language (zh-tw / zh-cn use BAAI/bge-base-zh-v1.5 at 768 dim; en uses
    paraphrase-multilingual-MiniLM-L12-v2 at 384 dim).

    - Max 100 texts per request
    - Returns normalised L2 embeddings
    """
    try:
        backend = get_backend()
        embeddings_list = backend.embed_batch(req.texts, req.language)
        model = _LANGUAGE_TO_MODEL[req.language]
        dims = _LANGUAGE_TO_DIM[req.language]
        return EmbedResponse(
            embeddings=embeddings_list,
            model=model,
            dimensions=dims,
        )
    except Exception as e:
        logger.exception("embedding_failed")
        raise HTTPException(status_code=500, detail=str(e))
