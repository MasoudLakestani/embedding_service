from sentence_transformers import SentenceTransformer

model_name = 'heydariAI/persian-embeddings'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed(text: str):
    model = get_model()
    embedding = model.encode(text)
    return embedding