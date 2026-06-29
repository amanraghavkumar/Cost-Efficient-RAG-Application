
import json
import pandas as pd
import numpy as np
import time
import os
from app.services.vector_store import vector_store_manager
from app.services.llm import llm_manager
from app.services.processor import document_processor

def calculate_metrics(results):
    hit_rates = []
    mrr = []
    recalls = []
    precisions = []
    ndcg = []

    for res in results:
        relevant = set(res['relevant_ids'])
        retrieved = res['retrieved_ids']
        
        hit = 1 if any(rid in relevant for rid in retrieved) else 0
        hit_rates.append(hit)
        
        rank = -1
        for i, rid in enumerate(retrieved):
            if rid in relevant:
                rank = i + 1
                break
        mrr.append(1.0 / rank if rank != -1 else 0)
        
        hits = len(relevant.intersection(set(retrieved)))
        recalls.append(hits / len(relevant) if len(relevant) > 0 else 0)
        precisions.append(hits / len(retrieved) if len(retrieved) > 0 else 0)
        
        dcg = 0.0
        for i, rid in enumerate(retrieved):
            if rid in relevant:
                dcg += 1.0 / np.log2(i + 2)
        idcg = 0.0
        for i in range(min(len(relevant), len(retrieved))):
            idcg += 1.0 / np.log2(i + 2)
        ndcg.append(dcg / idcg if idcg > 0 else 0)

    return {
        "Hit Rate": np.mean(hit_rates),
        "MRR": np.mean(mrr),
        "Recall": np.mean(recalls),
        "Context Precision": np.mean(precisions),
        "nDCG": np.mean(ndcg)
    }

def llm_judge(question, context, answer):
    prompt = f"""
    Evaluate the following RAG response.
    Question: {question}
    Context: {context}
    Answer: {answer}

    Provide scores from 0 to 1 for:
    1. Faithfulness: Is the answer derived solely from the context?
    2. Answer Relevance: Does the answer directly address the question?
    3. Groundedness: Is every claim in the answer supported by the context?

    Return result as JSON: {{"faithfulness": score, "relevance": score, "groundedness": score}}
    """
    try:
        # Wait to avoid rate limit
        time.sleep(20) 
        response = llm_manager.model.generate_content(prompt)
        text = response.text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        return json.loads(text)
    except Exception as e:
        print(f"Judge Error: {e}")
        return {"faithfulness": 0, "relevance": 0, "groundedness": 0}

def main():
    # 1. Ingest sample docs
    for file in os.listdir("data"):
        document_processor.process_and_ingest(f"data/{file}")

    # 2. Load dataset
    with open("evaluations/eval_dataset.json", "r") as f:
        dataset = json.load(f)

    retrieval_results = []
    answer_results = []

    print(f"Starting evaluation on {len(dataset)} samples. This will take some time due to API rate limits...")

    for i, item in enumerate(dataset):
        query = item['question']
        relevant_sources = item['relevant_chunk_ids']
        
        print(f"Processing question {i+1}/{len(dataset)}...")
        
        # Retrieve
        chunks = vector_store_manager.query(query, top_k=5)
        retrieved_ids = [c['metadata']['source'] for c in chunks]
        
        retrieval_results.append({
            "question": query,
            "relevant_ids": relevant_sources,
            "retrieved_ids": retrieved_ids
        })
        
        # Generate Answer
        try:
            time.sleep(20) # Wait to avoid rate limit
            llm_res = llm_manager.generate_answer(query, chunks)
            answer = llm_res['answer']
        except Exception as e:
            print(f"Generation Error: {e}")
            answer = "Error generating answer"
        
        context_text = "\n".join([c['text'] for c in chunks])
        scores = llm_judge(query, context_text, answer)
        
        answer_results.append({
            "question": query,
            "answer": answer,
            **scores
        })

    # 3. Calculate Retrieval Metrics
    metrics = calculate_metrics(retrieval_results)
    print("\n--- Retrieval Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 4. Save Results
    df_answers = pd.DataFrame(answer_results)
    df_answers.to_csv("evaluations/answer_eval_scores.csv", index=False)
    
    with open("evaluations/retrieval_report.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nEvaluation completed. Results saved to evaluations/ folder.")

if __name__ == "__main__":
    main()