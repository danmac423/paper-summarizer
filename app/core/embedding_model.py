from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from app.core.exceptions import EmbeddingError
from app.core.singleton_meta import SingletonMeta


class EmbeddingModel(metaclass=SingletonMeta):
    """
    Singleton class to manage the embedding model.
    This class ensures that the model is only initialized once,
    even if called from multiple threads.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initializes the embedding model.
        Args:
            model_name (str): Name of the embedding model.
            device (str): Device to run the model on (e.g., "cpu", "cuda").
        """
        self.model_name = model_name
        self.device = device
        self.model = self._initialize_model()

    def _initialize_model(self) -> HuggingFaceEmbeddings:
        """
        Initializes the embedding model.
        Returns:
            HuggingFaceEmbeddings: Initialized embedding model.
        Raises:
            EmbeddingError: If there is an error during model initialization.
        """
        model_kwargs = {"device": self.device}
        try:
            model = HuggingFaceEmbeddings(
                model_name=self.model_name, model_kwargs=model_kwargs
            )
            return model
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize embedding model {self.model_name} on {self.device}: {e}"
            ) from e
