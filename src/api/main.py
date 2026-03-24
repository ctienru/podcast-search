"""
Embedding API Server

Thin FastAPI layer exposing embedding encoder for external services (e.g., Java backend).

Usage:
    # Development
    uvicorn src.api.main:app --reload --port 8081

    # Production
    uvicorn src.api.main:app --host 0.0.0.0 --port 8081 --workers 1

API Endpoints:
    GET  /health  - Health check (warms up model)
    POST /embed   - Generate embeddings for texts
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.utils.logging import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model on startup (optional, remove for faster cold start)."""
    # Uncomment to pre-load model:
    # from src.api.routes import get_encoder
    # get_encoder()
    yield


app = FastAPI(
    title="Podcast Embedding API",
    description="Embedding service for podcast search",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow Java backend to call
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


