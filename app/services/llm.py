import google.generativeai as genai
from app.core.config import config
import time

class LLMManager:
    def __init__(self):
        if config.LLM_PROVIDER == "gemini":
            genai.configure(api_key=config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        elif config.LLM_PROVIDER == "openai":
            # Placeholder for OpenAI implementation if needed
            raise NotImplementedError("OpenAI integration not fully implemented in this version.")
        else:
            raise ValueError("No LLM provider configured. Please set GEMINI_API_KEY or OPENAI_API_KEY in .env")

    def generate_answer(self, query: str, context_chunks: list):
        # Build context string with chunk citations
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            # Using chunk_id from metadata for citations
            cid = chunk['metadata'].get('chunk_id', i)
            context_text += f"[Chunk {cid}]: {chunk['text']}\n\n"

        prompt = f"""
        You are a helpful assistant. Answer the following question strictly using the provided context.
        If the answer is not contained within the context, return exactly: "I could not find relevant information in the provided documents."
        Avoid hallucinations. Cite the chunk numbers used in your answer (e.g., [Chunk 1]).

        Context:
        {context_text}

        Question:
        {query}

        Answer:
        """
        
        start_time = time.time()
        response = self.model.generate_content(prompt)
        latency = time.time() - start_time
        
        # Gemini token usage is a bit different; for this assignment, we'll approximate or use usage_metadata if available
        token_usage = getattr(response, 'usage_metadata', None)
        
        return {
            "answer": response.text,
            "latency": latency,
            "token_usage": token_usage
        }

llm_manager = LLMManager()
