import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K = int(os.getenv("TOP_K", 5))
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")
    
    # Determine which LLM to use based on available key
    LLM_PROVIDER = "gemini" if GEMINI_API_KEY else "openai" if OPENAI_API_KEY else None

config = Config()
