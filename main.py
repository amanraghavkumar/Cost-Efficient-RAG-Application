import warnings
from cryptography.utils import CryptographyDeprecationWarning
from fastapi.middleware.cors import CORSMiddleware

# Suppress the specific CryptographyDeprecationWarning from pypdf
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

import time
import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel

from app.core.config import config
from app.services.processor import document_processor
from app.services.vector_store import vector_store_manager
from app.services.llm import llm_manager

# Setup logging
os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cost-Efficient RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods
    allow_headers=["*"], # Allow all headers
)
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = config.TOP_K
    filters: Optional[dict] = None

class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: List[dict]
    latency: float
    token_usage: Optional[dict]
    chunk_count: int

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/ingest")
async def ingest_documents(files: List[UploadFile] = File(..., description="Upload PDF, HTML or Markdown files")):
    try:
        total_chunks = 0
        for file in files:
            # Save file temporarily to disk
            temp_path = f"data/{file.filename}"
            os.makedirs("data", exist_ok=True)
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            
            chunks_count = document_processor.process_and_ingest(temp_path)
            total_chunks += chunks_count
            
            # --- YAHAN HUMNE LIVE LOGGING ADD KI HAI ---
            # Database ka current total count nikalna
            current_db_count = vector_store_manager.collection.count()
            print(f"\n [INGESTION EVENT]")
            print(f" File Uploaded: {file.filename}")
            print(f" Chunks added from this file: {chunks_count}")
            print(f"Total Vectors now in DB: {current_db_count}")
            print("-" * 20)
            # -------------------------------------------
            
            logger.info(f"Ingested {file.filename}: {chunks_count} chunks created.")
            
        return {"message": "Documents ingested successfully", "total_chunks": total_chunks}
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    try:
        start_time = time.time()
        
        # 1. Retrieval
        retrieval_start = time.time()
        chunks = vector_store_manager.query(
            query_text=request.query, 
            top_k=request.top_k, 
            filters=request.filters
        )
        retrieval_latency = time.time() - retrieval_start
        
        # 2. Generation
        llm_result = llm_manager.generate_answer(request.query, chunks)
        
        total_latency = time.time() - start_time
        
        # Logging
        logger.info(
            f"Query: {request.query} | "
            f"Retrieval Latency: {retrieval_latency:.4f}s | "
            f"Chunks Retrieved: {len(chunks)} | "
            f"Token Usage: {llm_result['token_usage']} | "
            f"Total Latency: {total_latency:.4f}s"
        )
        
        return QueryResponse(
            answer=llm_result['answer'],
            retrieved_chunks=chunks,
            latency=total_latency,
            token_usage=llm_result['token_usage'],
            chunk_count=len(chunks)
        )
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
