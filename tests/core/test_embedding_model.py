from app.core.embedding_model import EmbeddingModel, EmbeddingError
from app.core.singleton_meta import SingletonMeta

import pytest


@pytest.fixture(autouse=True)
def reset_singleton_meta_instances():
    with SingletonMeta._lock:
        SingletonMeta._instances = {}

    yield


def test_embedding_model_initializes_on_first_call(mocker):
    mock_huggingface_embeddings_class = mocker.patch(
        "app.core.embedding_model.HuggingFaceEmbeddings"
    )

    EmbeddingModel()
    mock_huggingface_embeddings_class.assert_called_once_with(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )


def test_embedding_model_returns_same_instance_on_subsequent_calls(mocker):
    mock_huggingface_embeddings_class = mocker.patch(
        "app.core.embedding_model.HuggingFaceEmbeddings"
    )

    model_1 = EmbeddingModel()
    model_2 = EmbeddingModel()
    model_3 = EmbeddingModel()

    mock_huggingface_embeddings_class.assert_called_once_with(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )

    assert model_1 is model_2 is model_3


def test_embedding_model_initialization_error(mocker):
    mock_huggingface_embeddings_class = mocker.patch(
        "app.core.embedding_model.HuggingFaceEmbeddings"
    )
    huggingface_initialization_error = RuntimeError(
        "Simulated error during model loading"
    )
    mock_huggingface_embeddings_class.side_effect = huggingface_initialization_error

    with pytest.raises(EmbeddingError) as excinfo:
        EmbeddingModel()

    assert "Failed to initialize embedding model" in str(excinfo.value)
    assert excinfo.value.__cause__ is huggingface_initialization_error
    mock_huggingface_embeddings_class.assert_called_once_with(
        model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}
    )
    assert not SingletonMeta._instances
