import os

from app.core.text_extractor import MarkerTextExtractor


RAW_DOCS_DIRECTORY = "data/raw/documents"
PROCESSED_DOCS_DIRECTORY = "data/processed/documents"

if __name__ == "__main__":
    os.makedirs(PROCESSED_DOCS_DIRECTORY, exist_ok=True)

    text_extractor = MarkerTextExtractor()

    for filename in os.listdir(RAW_DOCS_DIRECTORY):
        if filename.endswith(".pdf"):
            raw_file_path = os.path.join(RAW_DOCS_DIRECTORY, filename)
            processed_file_path = os.path.join(
                PROCESSED_DOCS_DIRECTORY, filename.replace(".pdf", ".md")
            )

            if not os.path.exists(processed_file_path):
                print(f"Processing {filename}...")
                article_text = text_extractor.extract_text_from_pdf_file(raw_file_path)
                with open(processed_file_path, "w") as f:
                    f.write(article_text)
                print(f"Processed {filename}")
            else:
                print(f"{filename} already exists. Skipping processing.")
