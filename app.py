from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Text Embedding Service",
    description="API for generating text embeddings using heydariAI/persian-embeddings",
    version="1.0.0"
)

model = None

@app.on_event("startup")
async def load_model():
    """Load the embedding model once at startup"""
    global model
    model_name = 'heydariAI/persian-embeddings'
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded successfully")

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed", min_length=1)

class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)

class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Text embedding vector")

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Text Embedding Service API",
        "endpoints": {
            "/embed": "POST - Embed single text",
            "/embed/batch": "POST - Embed multiple texts",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "heydariAI/persian-embeddings"}

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """
    Embed a single text

    Args:
        request: EmbedRequest containing the text to embed

    Returns:
        EmbedResponse with the embedding vector
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        embedding = model.encode(request.text)
        return EmbedResponse(embedding=embedding.tolist())

    except Exception as e:
        logger.error(f"Error embedding text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/embed/batch", response_model=EmbedBatchResponse)
async def embed_batch(request: EmbedBatchRequest):
    """
    Embed multiple texts in batch (more efficient than calling /embed multiple times)

    Args:
        request: EmbedBatchRequest containing list of texts to embed

    Returns:
        EmbedBatchResponse with list of embedding vectors
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        embeddings = model.encode(request.texts)
        return EmbedBatchResponse(embeddings=[emb.tolist() for emb in embeddings])

    except Exception as e:
        logger.error(f"Error embedding batch: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
