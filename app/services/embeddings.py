from sentence_transformers import SentenceTransformer
from app.core.config import config

class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str):
        return self.model.encode(text).tolist()

    def encode_batch(self, texts: list[str]):
        return self.model.encode(texts).tolist()

    def get_model_info(self):
        return {
            "model_name": config.EMBEDDING_MODEL_NAME,
            "dimension": self.dimension
        }

embedding_manager = EmbeddingManager()
