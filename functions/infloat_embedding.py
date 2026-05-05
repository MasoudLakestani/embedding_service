import os
from sentence_transformers import SentenceTransformer

os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

model_name = 'intfloat/multilingual-e5-small'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name, local_files_only=True)
    return _model

def embed_intfloat(text: str):
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding

def embed_intfloat_batch(texts: list):
    """Batch encode multiple texts efficiently"""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings