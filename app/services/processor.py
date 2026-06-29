import os
from pypdf import PdfReader
from bs4 import BeautifulSoup
from app.core.config import config
from app.services.vector_store import vector_store_manager

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext == '.html':
            return self._extract_html(file_path)
        elif ext == '.md':
            return self._extract_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def _extract_pdf(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_html(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text(separator=' ')

    def _extract_markdown(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def chunk_text(self, text: str, filename: str):
        # Simple character-based chunking with overlap
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            chunks.append({
                "id": vector_store_manager.generate_id(filename, chunk_id),
                "text": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_id": chunk_id,
                    "model_name": config.EMBEDDING_MODEL_NAME,
                    "dimension": 384 # all-MiniLM-L6-v2 dimension
                }
            })
            
            start += (self.chunk_size - self.chunk_overlap)
            chunk_id += 1
            
        return chunks

    def process_and_ingest(self, file_path: str):
        filename = os.path.basename(file_path)
        text = self.extract_text(file_path)
        chunks = self.chunk_text(text, filename)
        vector_store_manager.add_documents(chunks)
        return len(chunks)

document_processor = DocumentProcessor()
