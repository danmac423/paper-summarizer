from src.core.services.summary_service import generate_summary
from src.core.models.llm import get_chat_llm


import os

PROCESSED_DOCS_DIRECTORY = "data/processed/documents"
SUMMARIES_DIRECTORY = "data/processed/summaries"

MODELS = {
    "gpt-4.1-mini": {
        "api_key": "OPENAI_API_KEY",
        "model_name": "gpt-4.1-mini",
    },
    "gemini-2.0-flash-lite": {
        "api_key": "GEMINI_API_KEY",
        "model_name": "gemini-2.0-flash-lite",
    },
}

N_WORDS = 250


if __name__ == "__main__":
    os.makedirs(SUMMARIES_DIRECTORY, exist_ok=True)

    for model_name, config in MODELS.items():
        os.makedirs(os.path.join(SUMMARIES_DIRECTORY, model_name), exist_ok=True)

        API_KEY = config["api_key"]
        LLM_MODEL_NAME = config["model_name"]
        print(f"Using model: {LLM_MODEL_NAME}")

        llm = get_chat_llm(LLM_MODEL_NAME, API_KEY)

        for filename in os.listdir(PROCESSED_DOCS_DIRECTORY):
            if filename.endswith(".md"):
                doc_file_path = os.path.join(PROCESSED_DOCS_DIRECTORY, filename)
                with open(doc_file_path, "r") as f:
                    full_text = f.read()

                summary_file_path = os.path.join(
                    SUMMARIES_DIRECTORY, model_name, filename
                )
                if not os.path.exists(summary_file_path):
                    print(f"Generating summary for {filename}...")

                    summary = generate_summary(full_text, llm, N_WORDS)
                    with open(summary_file_path, "w") as f:
                        f.write(summary)
                    print(f"Generated summary for {filename}")
                else:
                    print(
                        f"Summary for {filename} already exists. Skipping generation."
                    )
                    continue

                print(f"Summary saved to {summary_file_path}")
