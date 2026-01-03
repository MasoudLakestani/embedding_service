from pydantic import BaseModel, Field
from typing import List

class EmbedRequest(BaseModel):
    text: str = Field(..., description="Text to embed", min_length=1)

class EmbedBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed", min_items=1)

class EmbedResponse(BaseModel):
    embedding: List[float] = Field(..., description="Text embedding vector")

class EmbedBatchResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
