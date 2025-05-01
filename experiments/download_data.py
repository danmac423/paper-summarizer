import requests
import os


DOCS_TO_DOWNLOAD = {
    "https://arxiv.org/pdf/1908.10084": "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.pdf",
    "https://arxiv.org/pdf/2004.04906": "Dense Passage Retrieval for Open-Domain Question Answering.pdf",
    "https://arxiv.org/pdf/1910.13461": "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.pdf",
    "https://arxiv.org/pdf/2106.09685": "LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS.pdf",
    "https://arxiv.org/pdf/1706.03762": "Attention Is All You Need.pdf",
}
DOCS_DIRECTORY = "data/raw/documents"


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename


if __name__ == "__main__":
    os.makedirs(DOCS_DIRECTORY, exist_ok=True)

    for url, filename in DOCS_TO_DOWNLOAD.items():
        local_path = os.path.join(DOCS_DIRECTORY, filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            download_file(url, local_path)
            print(f"Downloaded {filename}")
        else:
            print(f"{filename} already exists. Skipping download.")
