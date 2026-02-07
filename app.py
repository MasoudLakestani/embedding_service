from fastapi import FastAPI, HTTPException, Header
from typing import Optional
import logging
import asyncio
from schemas import EmbedRequest, EmbedBatchRequest, EmbedResponse, EmbedBatchResponse
from functions import (
    heydari_embedding,
    heydari_embedding_batch,
    get_heydari_model,
    embed_intfloat,
    embed_intfloat_batch,
    get_intfloat_model
)
import functions.heydari_embedding as heydari_module
import functions.infloat_embedding as intfloat_module

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Embedding Service",
    description="API for generating text embeddings using multiple models",
    version="1.0.0"
)

# Supported models mapping
SUPPORTED_MODELS = {
    "heydari": "heydariAI/persian-embeddings",
    "intfloat-small": "intfloat/multilingual-e5-small"
}

@app.on_event("startup")
async def load_models():
    """Load all embedding models once at startup"""
    logger.info("Loading models...")
    logger.info("Loading heydariAI/persian-embeddings")
    get_heydari_model()
    logger.info("Loading intfloat/multilingual-e5-small")
    get_intfloat_model()
    logger.info("All models loaded successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Text Embedding Service API",
        "supported_models": list(SUPPORTED_MODELS.keys()),
        "endpoints": {
            "/embed": "POST - Embed single text (requires 'model' header: heydari or intfloat-small)",
            "/embed/batch": "POST - Embed multiple texts (requires 'model' header: heydari or intfloat-small)",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_status = {
        "heydari": heydari_module._model is not None,
        "intfloat-small": intfloat_module._model is not None
    }

    if not all(models_status.values()):
        raise HTTPException(status_code=503, detail=f"Some models not loaded: {models_status}")

    return {
        "status": "healthy",
        "models": SUPPORTED_MODELS,
        "models_loaded": models_status
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest, model: Optional[str] = Header(None)):
    """
    Embed a single text

    Args:
        request: EmbedRequest containing the text to embed
        model: Model selection header (heydari or intfloat-small)

    Returns:
        EmbedResponse with the embedding vector
    """
    try:
        # Validate model header
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing 'model' header. Must be one of: {list(SUPPORTED_MODELS.keys())}"
            )

        if model not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{model}'. Must be one of: {list(SUPPORTED_MODELS.keys())}"
            )

        # Route to appropriate embedding function - run in thread pool for true async
        if model == "heydari":
            embedding = await asyncio.to_thread(heydari_embedding, request.text)
        elif model == "intfloat-small":
            embedding = await asyncio.to_thread(embed_intfloat, request.text)

        return EmbedResponse(embedding=embedding.tolist())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding text with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest, model: Optional[str] = Header(None)):
    """
    Embed multiple texts in batch (more efficient than calling /embed multiple times)

    Args:
        request: EmbedBatchRequest containing list of texts to embed
        model: Model selection header (heydari or intfloat-small)

    Returns:
        EmbedBatchResponse with list of embedding vectors
    """
    try:
        # Validate model header
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=f"Missing 'model' header. Must be one of: {list(SUPPORTED_MODELS.keys())}"
            )

        if model not in SUPPORTED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model '{model}'. Must be one of: {list(SUPPORTED_MODELS.keys())}"
            )

        # Route to appropriate embedding function - using batch processing in thread pool
        if model == "heydari":
            embeddings = await asyncio.to_thread(heydari_embedding_batch, request.texts)
        elif model == "intfloat-small":
            embeddings = await asyncio.to_thread(embed_intfloat_batch, request.texts)

        return EmbedBatchResponse(embeddings=embeddings.tolist())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error embedding batch with model '{model}': {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
