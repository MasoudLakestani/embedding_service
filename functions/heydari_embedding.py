from sentence_transformers import SentenceTransformer

model_name = 'heydariAI/persian-embeddings'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def heydari_embedding(text: str):
    model = get_model()
    embedding = model.encode(text)
    return embedding

def heydari_embedding_batch(texts: list):
    """Batch encode multiple texts efficiently"""
    model = get_model()
    embeddings = model.encode(texts)
    return embeddings