import os
from rag_engine import ProjectRAGEngine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

# 1. Define your Golden Dataset
# Replace 'your-project-id' and the expected_source with actual data from your workspace
PROJECT_ID = "7e0da45d" 
GOLDEN_DATASET = [
    {
        "question": "What is the primary evidence for Spinosaurus being aquatic?",
        "expected_source": "spinosaurus_nature_2020.pdf", # The exact filename or OpenAlex title
        "ground_truth": "High bone density and a paddle-like tail suggest an aquatic lifestyle."
    },
    {
        "question": "When did the T-Rex go extinct?",
        "expected_source": "cretaceous_extinction_overview",
        "ground_truth": "The T-Rex went extinct approximately 66 million years ago during the K-Pg extinction event."
    }
]

def run_traditional_eval(engine, k=5):
    print(f"--- Running Traditional Metrics (Top {k}) ---")
    retriever = engine.get_vectorstore().as_retriever(search_kwargs={"k": k})
    
    hits = 0
    mrr_sum = 0
    
    for item in GOLDEN_DATASET:
        docs = retriever.invoke(item["question"])
        
        # Check metadata for filename (PDFs) or title (OpenAlex)
        retrieved_sources = [
            d.metadata.get("filename") or d.metadata.get("title") for d in docs
        ]
        
        expected = item["expected_source"]
        
        # Hit Rate (Recall@k): Was the right document retrieved at all?
        if expected in retrieved_sources:
            hits += 1
            # MRR: How high up was it? (1/rank)
            rank = retrieved_sources.index(expected) + 1
            mrr_sum += 1 / rank
            print(f"✅ {item['question'][:30]}... -> Found at Rank {rank}")
        else:
            print(f"❌ {item['question'][:30]}... -> Not found in top {k}")
            
    print(f"Hit Rate (Recall@{k}): {hits / len(GOLDEN_DATASET):.2f}")
    print(f"Mean Reciprocal Rank (MRR): {mrr_sum / len(GOLDEN_DATASET):.2f}\n")

def run_ragas_eval(engine):
    print("--- Running Ragas Framework Evaluation ---")
    retriever = engine.get_vectorstore().as_retriever(search_kwargs={"k": 5})
    
    data_samples = {
        "question": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for item in GOLDEN_DATASET:
        # Fetch contexts
        docs = retriever.invoke(item["question"])
        contexts = [d.page_content for d in docs]
        
        data_samples["question"].append(item["question"])
        data_samples["contexts"].append(contexts)
        data_samples["ground_truth"].append(item["ground_truth"])
        
    dataset = Dataset.from_dict(data_samples)
    
    # Run Ragas evaluation using your existing Gemini LLM and Embeddings
    results = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=engine.llm,
        embeddings=engine.embeddings
    )
    
    print(results)

if __name__ == "__main__":
    # Initialize your RAG engine
    engine = ProjectRAGEngine(PROJECT_ID)
    
    run_traditional_eval(engine, k=5)
    run_ragas_eval(engine)