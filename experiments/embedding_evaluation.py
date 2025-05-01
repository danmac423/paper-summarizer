import json
import os
from typing import Dict, List, Any

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


# --- Configuration ---
CHUNKED_DOCS_DIRECTORY = "data/processed/chunks"
EMBEDDING_DATASET_DIRECTORY = "data/embedding_dataset"
EVAL_RESULTS_DIRECTORY = "results"


MODEL_NAMES_TO_EVALUATE = [
    "Alibaba-NLP/gte-multilingual-base",
    "intfloat/multilingual-e5-large-instruct",
    "Lajavaness/bilingual-embedding-large",
]

MODEL_KWARGS = {
    "device": "mps",  # Change to 'cuda' or 'cpu' depending on your hardware
    "trust_remote_code": True,
}

RECALL_KS_TO_EVALUATE = [
    1,
    3,
    5,
    10,
]  # Calculate Recall@1, Recall@3, Recall@5, Recall@10

K_RETRIEVAL_SEARCH_LIMIT = (
    max(RECALL_KS_TO_EVALUATE) + 5 if RECALL_KS_TO_EVALUATE else 1
)


# --- Helper Functions (Metrics) ---


def calculate_mrr(
    ground_truth_chunk_ids: List[str], ranked_chunk_ids: List[str]
) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) for a single query.
    Returns 1.0/(rank of first relevant) or 0.0 if no relevant found in rank.
    Rank is 1-based for MRR calculation.
    Assumes ground_truth_chunk_ids is not empty for questions where a relevant chunk exists.
    """
    for rank, chunk_id in enumerate(ranked_chunk_ids):
        if chunk_id in ground_truth_chunk_ids:
            return 1.0 / (rank + 1)
    return 0.0


def calculate_recall(
    ground_truth_chunk_ids: List[str], ranked_chunk_ids: List[str]
) -> float:
    """
    Calculates Recall for a single query based on a list of ranked chunk IDs (top K).
    K is implicitly the length of ranked_chunk_ids.
    Assumes ground_truth_chunk_ids is not empty for questions where a relevant chunk exists.
    """
    relevant_count = sum(
        1 for chunk_id in ranked_chunk_ids if chunk_id in ground_truth_chunk_ids
    )
    return relevant_count / len(ground_truth_chunk_ids)


# --- Data Loading Functions ---


def load_chunks(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads chunks from JSONL files in the given directory."""
    print(f"Loading chunks from: {directory}")
    chunks_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                chunks_list = [json.loads(line) for line in f]
            chunks_data[filename] = chunks_list
            print(f"  Loaded {len(chunks_list)} chunks from {filename}")
    return chunks_data


def load_questions(directory: str) -> Dict[str, List[Dict[str, Any]]]:
    """Loads questions and ground truth from JSONL files in the given directory."""
    print(f"Loading questions from: {directory}")
    questions_answers_data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                qa_list = [json.loads(line) for line in f]
            questions_answers_data[filename] = qa_list
            print(f"  Loaded {len(qa_list)} questions from {filename}")
    return questions_answers_data


# --- Vector Store Creation Function ---


def create_vector_stores(
    chunks_data: Dict[str, List[Dict[str, Any]]], embedding_model: HuggingFaceEmbeddings
) -> Dict[str, FAISS]:
    """Creates and returns a dictionary of FAISS vector stores (one per source chunk file)."""
    print(f"\nCreating FAISS vector stores for model: {embedding_model.model_name}")
    vector_stores = {}

    for filename, chunks_list in chunks_data.items():
        print(
            f"  Creating FAISS store for {filename} with {len(chunks_list)} chunks..."
        )
        vector_store = FAISS.from_texts(
            texts=[chunk["text"] for chunk in chunks_list],
            embedding=embedding_model,
            metadatas=[{"id": chunk["id"]} for chunk in chunks_list],
        )
        vector_stores[filename] = vector_store
        print(f"  FAISS store for {filename} created.")

    return vector_stores


# --- Evaluation Function ---


def evaluate_model_on_dataset(
    vector_stores: Dict[str, FAISS],
    questions_answers_data: Dict[str, List[Dict[str, Any]]],
    k_retrieval_search_limit: int,
    recall_ks: List[int],
) -> Dict[str, Any]:
    """
    Evaluates the embedding model (reprezented by the vector_stores)
    based on questions and ground truth.
    Performs search up to k_retrieval_search_limit.
    Calculates MRR (on full search results) and Recall@K for specified Ks.
    Returns a dictionary with average metrics (MRR and Recall@K for specified Ks).
    Assumes data is well-formed and searches succeed.
    """
    print(
        f"\nStarting evaluation (Retrieval Search Limit K={k_retrieval_search_limit})..."
    )

    all_mrr_scores: List[float] = []
    all_recall_scores_at_k: Dict[int, List[float]] = {k: [] for k in recall_ks}
    evaluated_questions_count = 0

    for filename, faiss_index in vector_stores.items():
        qa_list = questions_answers_data[filename]

        for q_entry in qa_list:
            question_text = q_entry["question"]
            ground_truth_chunk_ids = q_entry["chunk_ids"]

            search_results = faiss_index.similarity_search(
                query=question_text, k=k_retrieval_search_limit
            )

            ranked_chunk_ids = [result.metadata["id"] for result in search_results]

            mrr = calculate_mrr(ground_truth_chunk_ids, ranked_chunk_ids)
            all_mrr_scores.append(mrr)

            for k in recall_ks:
                current_k = min(k, len(ranked_chunk_ids))

                top_k_ranked_ids = ranked_chunk_ids[:current_k]
                recall_at_k = calculate_recall(ground_truth_chunk_ids, top_k_ranked_ids)

                all_recall_scores_at_k[k].append(recall_at_k)

            evaluated_questions_count += 1

    mean_mrr = np.mean(all_mrr_scores)

    mean_recalls_at_k = {
        k: np.mean(scores) for k, scores in all_recall_scores_at_k.items()
    }

    print("\nEvaluation finished.")
    print(f"Number of questions evaluated: {evaluated_questions_count}")

    return {
        "mean_mrr": mean_mrr,
        "mean_recalls_at_k": mean_recalls_at_k,
        "k_retrieval_search_limit": k_retrieval_search_limit,
        "evaluated_questions_count": evaluated_questions_count,
    }


# --- Main Execution Block ---

if __name__ == "__main__":
    all_chunks_data = load_chunks(CHUNKED_DOCS_DIRECTORY)
    all_questions_answers_data = load_questions(EMBEDDING_DATASET_DIRECTORY)

    all_models_evaluation_results: Dict[str, Dict[str, Any]] = {}

    for model_name in MODEL_NAMES_TO_EVALUATE:
        print(f"\n{'-' * 70}")
        print(f"STARTING EVALUATION FOR MODEL: {model_name}")
        print(f"{'-' * 70}")

        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=MODEL_KWARGS
        )
        print(f"Model '{model_name}' loaded successfully.")

        vector_stores_for_model = create_vector_stores(all_chunks_data, embedding_model)

        model_evaluation_metrics = evaluate_model_on_dataset(
            vector_stores_for_model,
            all_questions_answers_data,
            k_retrieval_search_limit=K_RETRIEVAL_SEARCH_LIMIT,
            recall_ks=RECALL_KS_TO_EVALUATE,
        )

        all_models_evaluation_results[model_name] = model_evaluation_metrics

    eval_results_file = os.path.join(
        EVAL_RESULTS_DIRECTORY, "embedding_model_evaluation_results.json"
    )
    os.makedirs(EVAL_RESULTS_DIRECTORY, exist_ok=True)

    with open(eval_results_file, "w") as f:
        json.dump(all_models_evaluation_results, f, indent=4)
    print(f"\nEvaluation results saved to: {eval_results_file}")

    
    print(f"\n\n{'=' * 70}")
    print("EMBEDDING MODEL EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    sorted_results = sorted(
        all_models_evaluation_results.items(),
        key=lambda item: item[1]["mean_mrr"],
        reverse=True,
    )

    col_width_model = 60
    col_width_questions = 15
    col_width_mrr = 15
    col_width_recall_col = 15

    recall_header_cols = [f"Recall@{k}" for k in RECALL_KS_TO_EVALUATE]
    header_row_list = ["Model", "Qs Eval", "Mean MRR"] + recall_header_cols

    header_line = f"{header_row_list[0]:<{col_width_model}} | {header_row_list[1]:<{col_width_questions}} | {header_row_list[2]:<{col_width_mrr}} | {' | '.join([f'{h:<{col_width_recall_col}}' for h in header_row_list[3:]])}"
    print(header_line)

    fixed_line_part = f"{'-' * col_width_model}-|-{'-' * col_width_questions}-|-{'-' * col_width_mrr}-|-"
    recall_line_parts = [f"{'-' * col_width_recall_col}" for _ in RECALL_KS_TO_EVALUATE]
    dynamic_line_part = " | ".join(recall_line_parts)
    full_separator_line = fixed_line_part + dynamic_line_part
    print(full_separator_line)

    for model_name, metrics in sorted_results:
        recall_values_formatted_list = [
            f"{metrics['mean_recalls_at_k'][k]:<{col_width_recall_col}.4f}"
            for k in RECALL_KS_TO_EVALUATE
        ]
        recall_values_formatted_str = " | ".join(recall_values_formatted_list)

        print(
            f"{model_name:<{col_width_model}} | "
            f"{metrics['evaluated_questions_count']:<{col_width_questions}} | "
            f"{metrics['mean_mrr']:.4f}{'':<{col_width_mrr - 5}} | "
            f"{recall_values_formatted_str}"
        )

    print(f"\n{'=' * 70}")
    print("END OF EVALUATION")
    print(f"{'=' * 70}")
