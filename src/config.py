import torch


torch.classes.__path__ = []

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEPARATORS = ["\n\n", "\n", " ", ""]
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

K_RETRIEVED_DOCS = 4

EMBEDDING_MODEL_NAME = "Lajavaness/bilingual-embedding-large"
