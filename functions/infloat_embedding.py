from sentence_transformers import SentenceTransformer

model_name = 'intfloat/multilingual-e5-small'
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model

def embed_intfloat(text: str):
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding