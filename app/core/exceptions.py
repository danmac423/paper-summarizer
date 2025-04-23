class LLMServiceError(Exception):
    """Custom exception for LLM service errors."""

    pass


class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""

    pass


class EmbeddingError(Exception):
    """Custom exception for embedding generation errors."""

    pass


class VectorStoreError(Exception):
    """Custom exception for vector store operations (FAISS)."""

    pass


class RAGServiceError(Exception):
    """Custom exception for RAG chain errors."""

    pass


class SummaryServiceError(Exception):
    """Custom exception for summarization errors."""

    pass
