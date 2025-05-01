import os
import json
from bert_score import score as bert_score
import evaluate
import torch

# --- Configuration ---
SUMMARY_DATASET_DIRECTORY = "data/summary_dataset"
SUMMARIES_DIRECTORY = "data/processed/summaries"
RESULTS_DIRECTORY = "results"
RESULTS_FILE = os.path.join(RESULTS_DIRECTORY, "summary_evaluation_results.json")
BERT_SCORE_LANG = "en"


def load_reference_summaries(directory):
    references = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as file:
                references[filename] = file.read()
    return references


def calculate_summary_scores(
    summary_text,
    reference_text,
    rouge_scorer,
    bleu_scorer,
    bert_score_func,
    bert_lang,
    device,
):
    scores = {}

    rouge_scores = rouge_scorer.compute(
        predictions=[summary_text], references=[reference_text]
    )
    scores["rouge1_fmeasure"] = rouge_scores["rouge1"]
    scores["rouge2_fmeasure"] = rouge_scores["rouge2"]
    scores["rougeL_fmeasure"] = rouge_scores["rougeL"]
    scores["rougeLsum_fmeasure"] = rouge_scores["rougeLsum"]

    bleu_scores = bleu_scorer.compute(
        predictions=[summary_text], references=[[reference_text]]
    )
    scores["bleu"] = bleu_scores["bleu"]

    P, R, F1 = bert_score_func(
        [summary_text], [reference_text], lang=bert_lang, device=device
    )
    scores["bertscore_p"] = P.item()
    scores["bertscore_r"] = R.item()
    scores["bertscore_f1"] = F1.item()

    return scores


def compute_average_scores(scores_accumulator):
    average_scores = {}
    for metric, scores_list in scores_accumulator.items():
        if scores_list:
            average_scores[f"average_{metric}"] = sum(scores_list) / len(scores_list)
    return average_scores


def evaluate_model(
    model_name,
    summaries_directory,
    references,
    rouge_scorer,
    bleu_scorer,
    bert_score_func,
    bert_lang,
    device,
):
    print(f"\nProcessing model: {model_name}")
    model_summaries_path = os.path.join(summaries_directory, model_name)
    summary_filenames = [
        f
        for f in os.listdir(model_summaries_path)
        if os.path.isfile(os.path.join(model_summaries_path, f))
    ]

    if not summary_filenames:
        print(f"No summary files found for model {model_name}. Skipping.")
        return {}

    scores_accumulator = {
        "rouge1_fmeasure": [],
        "rouge2_fmeasure": [],
        "rougeL_fmeasure": [],
        "rougeLsum_fmeasure": [],
        "bleu": [],
        "bertscore_p": [],
        "bertscore_r": [],
        "bertscore_f1": [],
    }

    processed_count = 0
    for filename in summary_filenames:
        if filename not in references:
            print(
                f"Warning: No reference found for file {filename} for model {model_name}. Skipping."
            )
            continue

        summary_filepath = os.path.join(model_summaries_path, filename)
        with open(summary_filepath, "r", encoding="utf-8") as file:
            summary_text = file.read()

        reference_text = references[filename]

        summary_scores = calculate_summary_scores(
            summary_text,
            reference_text,
            rouge_scorer,
            bleu_scorer,
            bert_score_func,
            bert_lang,
            device,
        )

        for metric, score in summary_scores.items():
            scores_accumulator[metric].append(score)

        processed_count += 1

    print(f"Finished processing {processed_count} summaries for model {model_name}.")

    if processed_count > 0:
        average_scores = compute_average_scores(scores_accumulator)
        print(f"Average scores for {model_name}: {average_scores}")
        return average_scores
    else:
        print(
            f"No summaries successfully processed for {model_name}. No averages calculated."
        )
        return {}


def main():
    rouge_scorer = evaluate.load("rouge")
    bleu_scorer = evaluate.load("bleu")

    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device} for BERTScore.")

    print(f"Loading reference summaries from {SUMMARY_DATASET_DIRECTORY}...")
    references = load_reference_summaries(SUMMARY_DATASET_DIRECTORY)
    print(f"Loaded {len(references)} reference summaries.")

    all_model_average_results = {}

    model_directories = [
        d
        for d in os.listdir(SUMMARIES_DIRECTORY)
        if os.path.isdir(os.path.join(SUMMARIES_DIRECTORY, d))
    ]
    print(
        f"Found {len(model_directories)} model directories: {', '.join(model_directories)}"
    )

    for model_name in model_directories:
        model_avg_scores = evaluate_model(
            model_name,
            SUMMARIES_DIRECTORY,
            references,
            rouge_scorer,
            bleu_scorer,
            bert_score,
            BERT_SCORE_LANG,
            device,
        )
        all_model_average_results[model_name] = model_avg_scores

    print(f"\nSaving evaluation results to {RESULTS_FILE}...")
    os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as outfile:
        json.dump(all_model_average_results, outfile, indent=4)

    print("Results saved successfully.")
    print("Evaluation process finished.")


if __name__ == "__main__":
    main()
