import chromadb
from chromadb.config import Settings
from app.core.config import config
from app.services.embeddings import embedding_manager
import hashlib

class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name="rag_collection",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Record embedding info in a special metadata entry or just keep in config
        self.model_info = embedding_manager.get_model_info()

    def generate_id(self, filename: str, chunk_id: int):
        # Unique ID based on filename and chunk index
        return hashlib.sha256(f"{filename}_{chunk_id}".encode()).hexdigest()

    def add_documents(self, documents: list[dict]):
        """
        documents: list of {'text': str, 'metadata': dict}
        """
        texts = [doc['text'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [doc['id'] for doc in documents]
        
        # Embeddings are handled by ChromaDB if we don't provide them, 
        # but we use our custom SentenceTransformer here for consistency.
        embeddings = embedding_manager.encode_batch(texts)
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

    def query(self, query_text: str, top_k: int = config.TOP_K, filters: dict = None):
        query_embedding = embedding_manager.encode(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "score": results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results

vector_store_manager = VectorStoreManager()
