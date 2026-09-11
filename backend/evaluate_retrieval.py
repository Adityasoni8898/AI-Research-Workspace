import os
import json
import pandas as pd
from tqdm import tqdm
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from rag_engine import ProjectRAGEngine

load_dotenv()

# ==========================================
# Configuration
# ==========================================
PROJECT_ID = "59687e81"
DATASET_PATH = "Spinosaurid_golden_dataset.csv"
OUTPUT_PATH = "Spinosaurid_evaluation_results.csv"

# ==========================================
# LLM Judge Setup
# ==========================================
api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.0,
    google_api_key=api_key
)

class EvaluationScore(BaseModel):
    correctness_score: int = Field(
        description="Score from 1 to 5 evaluating accuracy and alignment with ground truth."
    )
    citation_score: int = Field(
        description="Score from 1 to 5 evaluating if the answer cited sources/papers properly."
    )
    reasoning: str = Field(
        description="Brief explanation justifying the scores."
    )

judge_parser = JsonOutputParser(pydantic_object=EvaluationScore)

judge_prompt = ChatPromptTemplate.from_template("""
You are an expert paleontology evaluator. Evaluate the generated RAG response against the ground truth answer.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Response: {generated_answer}

Evaluation criteria:
1. **correctness_score** (1 to 5):
   - 5: Perfectly accurate and aligns fully with ground truth.
   - 4: Mostly accurate with minor omissions.
   - 3: Partially correct, misses key details or contains mild inaccuracies.
   - 2: Mostly incorrect or unhelpful.
   - 1: Completely incorrect, misleading, or hallucinated.

2. **citation_score** (1 to 5):
   - 5: Explicitly cites relevant sources, paper titles, DOIs, or filenames.
   - 3: Gives general claims without distinct citations, or cites general knowledge.
   - 1: Fails to cite any source when making explicit claims.

3. **reasoning**: A concise 1-2 sentence explanation for the ratings.

{format_instructions}
""")

judge_chain = judge_prompt | judge_llm | judge_parser

# ==========================================
# Main Evaluation Loop
# ==========================================
def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset '{DATASET_PATH}' not found!")
        return

    print(f"Loading golden dataset: {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)

    print(f"Initializing RAG Engine for project ID: {PROJECT_ID}...")
    engine = ProjectRAGEngine(project_id=PROJECT_ID)

    results = []

    print("\nStarting evaluation run...\n")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating Questions"):
        q_id = row.get("id", "")
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth_answer", "")).strip()
        q_type = str(row.get("question_type", "General"))
        difficulty = str(row.get("difficulty", "Medium"))
        source_location = str(row.get("source_location", ""))
        notes = str(row.get("notes", ""))

        # 1. Generate answer using ProjectRAGEngine stream
        generated_tokens = []
        try:
            for token in engine.chat_stream(question):
                generated_tokens.append(token)
            generated_answer = "".join(generated_tokens)
        except Exception as e:
            generated_answer = f"ERROR: Engine query failed: {str(e)}"

        # 2. Run LLM Judge Evaluation
        try:
            judge_res = judge_chain.invoke({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "format_instructions": judge_parser.get_format_instructions()
            })
            c_score = judge_res.get("correctness_score", 0)
            cit_score = judge_res.get("citation_score", 0)
            reasoning = judge_res.get("reasoning", "")
        except Exception as e:
            c_score = 0
            cit_score = 0
            reasoning = f"Evaluation failed: {str(e)}"

        results.append({
            "id": q_id,
            "question_type": q_type,
            "difficulty": difficulty,
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": generated_answer,
            "correctness_score": c_score,
            "citation_score": cit_score,
            "judge_reasoning": reasoning,
            "source_location": source_location,
            "notes": notes
        })

    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved detailed evaluation results to: {OUTPUT_PATH}")

    # Display Breakdown Reports
    print("\n" + "="*60)
    print("                EVALUATION SUMMARY REPORT                ")
    print("="*60)
    
    overall_corr = results_df["correctness_score"].mean()
    overall_cit = results_df["citation_score"].mean()
    print(f"Overall Correctness Score: {overall_corr:.2f} / 5.0")
    print(f"Overall Citation Score:    {overall_cit:.2f} / 5.0")
    print("-" * 60)

    print("\n--- Breakdown by Question Type ---")
    type_summary = results_df.groupby("question_type")[["correctness_score", "citation_score"]].mean()
    print(type_summary.round(2).to_string())

    print("\n--- Breakdown by Difficulty ---")
    diff_summary = results_df.groupby("difficulty")[["correctness_score", "citation_score"]].mean()
    print(diff_summary.round(2).to_string())
    print("="*60 + "\n")

if __name__ == "__main__":
    run_evaluation()