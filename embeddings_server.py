#!/usr/bin/env python3
"""
Embeddings FastAPI Server

OpenAI-compatible Embeddings API server using sentence-transformers.
Single-file, local inference - no external API calls.
Default model: all-MiniLM-L6-v2 (384-dim, 22.7M params).
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Server Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
_HOST = os.getenv("EMBEDDINGS_HOST", "0.0.0.0")
_PORT = int(os.getenv("EMBEDDINGS_PORT", "8882"))
_MAX_CONCURRENT = int(os.getenv("EMBEDDINGS_MAX_CONCURRENT", "4"))
_DEFAULT_MODEL = os.getenv(
    "EMBEDDINGS_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
_EMBEDDING_DIMS = int(os.getenv("EMBEDDINGS_DIMS", "0"))  # 0 = use model default


def _normalize_input(texts: str | list[str]) -> list[str]:
    """Normalize input to a list of strings."""
    if isinstance(texts, str):
        return [texts]
    return [str(t) for t in texts]


def _run_embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Run embedding inference synchronously (called in thread pool)."""
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup. All state lives on app.state - no globals."""
    app.state.started_at = time.time()
    print(f"Loading embedding model '{_DEFAULT_MODEL}'...")
    app.state.model = SentenceTransformer(_DEFAULT_MODEL)
    dims = app.state.model.get_sentence_embedding_dimension()
    app.state.embedding_dims = _EMBEDDING_DIMS if _EMBEDDING_DIMS > 0 else dims
    app.state.executor = ThreadPoolExecutor(max_workers=_MAX_CONCURRENT)
    app.state.ready = True
    print(
        f"Embeddings ready! (model: {_DEFAULT_MODEL}, "
        f"dims: {app.state.embedding_dims}, "
        f"max {_MAX_CONCURRENT} concurrent)"
    )
    yield
    app.state.ready = False
    app.state.executor.shutdown(wait=False)


app = FastAPI(
    title="Embeddings API",
    description="OpenAI-compatible Embeddings API using sentence-transformers",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    model: str = _DEFAULT_MODEL
    input: str | list[str]
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: list[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: dict


@app.post("/v1/embeddings")
async def create_embedding(request: Request, req: EmbeddingRequest):
    """
    OpenAI-compatible embeddings endpoint.
    Accepts single string or list of strings.
    Returns 384-dim normalized embeddings from all-MiniLM-L6-v2.
    """
    if not req.input:
        raise HTTPException(400, "input is required")

    texts = _normalize_input(req.input)

    state = request.app.state
    loop = asyncio.get_running_loop()

    embeddings = await loop.run_in_executor(
        state.executor,
        _run_embed,
        state.model,
        texts,
    )

    data = [
        EmbeddingData(
            index=i,
            embedding=emb.tolist(),
        )
        for i, emb in enumerate(embeddings)
    ]

    return EmbeddingResponse(
        object="list",
        data=data,
        model=_DEFAULT_MODEL,
        usage={
            "prompt_tokens": sum(len(t.split()) for t in texts),
            "total_tokens": sum(len(t.split()) for t in texts),
        },
    )


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models endpoint. Returns the currently loaded model."""
    return {
        "object": "list",
        "data": [
            {
                "id": _DEFAULT_MODEL,
                "object": "model",
                "created": 0,
                "owned_by": "sentence-transformers",
                "description": (
                    f"Embedding model: {_DEFAULT_MODEL} "
                    f"({app.state.embedding_dims}-dim)"
                ),
            }
        ],
    }


@app.get("/health")
async def health(request: Request):
    """
    Health check endpoint. Always public - no auth required.
    Returns model loading state, uptime, memory usage, and server config.
    """
    state = request.app.state
    started_at = getattr(state, "started_at", None)
    uptime = int(time.time() - started_at) if started_at else 0
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        "status": "ok",
        "state": "ready" if getattr(state, "ready", False) else "loading",
        "model": _DEFAULT_MODEL,
        "embedding_dims": getattr(state, "embedding_dims", 0),
        "max_concurrent": _MAX_CONCURRENT,
        "uptime_seconds": uptime,
        "memory_rss_mb": mem_info.rss // (1024 * 1024),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=_HOST, port=_PORT, workers=1)
