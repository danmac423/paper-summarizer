from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TextSplitter


from app.core.embedding_model import EmbeddingModel
from app.core.exceptions import VectorStoreError


class DocumentProcessor:
    def __init__(self, embedding_model: EmbeddingModel, text_splitter: TextSplitter):
        self.embedding_model = embedding_model
        self.text_splitter = text_splitter

    def split_text(self, text: str) -> list[str]:
        """
        Splits the text into smaller chunks.
        Args:
            text (str): The text to be split.
        Returns:
            list[str]: List of text chunks.
        """
        return self.text_splitter.split_text(text)

    def create_vector_store(self, chunks: list[str]) -> FAISS:
        """
        Creates a FAISS vector store from the text chunks.
        Args:
            chunks (list[str]): List of text chunks.
        Returns:
            FAISS: FAISS vector store.
        Raises:
            VectorStoreError: If there is an error during vector store creation.
        """
        if not chunks:
            raise VectorStoreError("Cannot create vector store from empty chunks.")
        try:
            vector_store = FAISS.from_texts(
                texts=chunks, embedding=self.embedding_model.model
            )
            return vector_store
        except Exception as e:
            raise VectorStoreError(
                f"Error during FAISS vector store creation: {e}"
            ) from e
