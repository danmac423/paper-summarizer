import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from experiments.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SEPARATORS,
    EMBEDDING_MODEL_NAME,
)

PROCESSED_DOCS_DIRECTORY = "data/processed/documents"
CHUNKED_DOCS_DIRECTORY = "data/processed/chunks"


def save_chunks_to_jsonl(chunks, output_file_path):
    print(f"Saving chunks to {output_file_path}...")
    with open(output_file_path, "w") as f:
        for i, chunk in enumerate(chunks):
            data = {
                "id": i,
                "text": chunk,
            }
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":

    os.makedirs(CHUNKED_DOCS_DIRECTORY, exist_ok=True)
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    for filename in os.listdir(PROCESSED_DOCS_DIRECTORY):
        if filename.endswith(".md"):
            processed_file_path = os.path.join(PROCESSED_DOCS_DIRECTORY, filename)
            chunked_file_path = os.path.join(
                CHUNKED_DOCS_DIRECTORY, filename.replace(".md", ".jsonl")
            )

            if not os.path.exists(chunked_file_path):
                print(f"Chunking {filename}...")
                with open(processed_file_path, "r") as f:
                    article_text = f.read()
                chunks = text_splitter.split_text(article_text)
                save_chunks_to_jsonl(chunks, chunked_file_path)
                print(f"Chunked {filename}")
            else:
                print(f"{filename} already exists. Skipping chunking.")
