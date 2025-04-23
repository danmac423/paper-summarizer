import pytest
from unittest.mock import MagicMock

from app.core.document_processor import DocumentProcessor
from app.core.embedding_model import EmbeddingModel
from app.core.exceptions import VectorStoreError
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import TextSplitter


@pytest.fixture
def document_processor_with_mocks(mocker):
    mock_embedding_model_instance = MagicMock(spec=EmbeddingModel)
    mock_embedding_model_instance.model = MagicMock()

    mock_text_splitter_instance = MagicMock(spec=TextSplitter)

    processor = DocumentProcessor(
        embedding_model=mock_embedding_model_instance,
        text_splitter=mock_text_splitter_instance,
    )

    return processor, mock_embedding_model_instance, mock_text_splitter_instance


def test_document_processor_initializes_with_dependencies(
    document_processor_with_mocks,
):
    processor, mock_embedding_model, mock_text_splitter = document_processor_with_mocks

    assert processor.embedding_model is mock_embedding_model
    assert processor.text_splitter is mock_text_splitter


def test_document_processor_split_text_calls_splitter(document_processor_with_mocks):
    processor, _, mock_text_splitter = document_processor_with_mocks

    input_text = "This is a sample text to be split."
    expected_chunks = ["This is a sample", "text to be split."]

    mock_text_splitter.split_text.return_value = expected_chunks

    actual_chunks = processor.split_text(input_text)

    mock_text_splitter.split_text.assert_called_once_with(input_text)

    assert actual_chunks == expected_chunks


def test_document_processor_create_vector_store_succeeds(
    document_processor_with_mocks, mocker
):
    processor, mock_embedding_model, _ = document_processor_with_mocks

    mock_faiss_from_texts = mocker.patch("app.core.document_processor.FAISS.from_texts")

    mock_vector_store_instance = MagicMock(spec=FAISS)
    mock_faiss_from_texts.return_value = mock_vector_store_instance

    input_chunks = ["chunk 1", "chunk 2", "chunk 3"]

    actual_vector_store = processor.create_vector_store(input_chunks)

    mock_faiss_from_texts.assert_called_once_with(
        texts=input_chunks,
        embedding=mock_embedding_model.model,
    )

    assert actual_vector_store is mock_vector_store_instance


def test_document_processor_create_vector_store_raises_error_on_empty_chunks(
    document_processor_with_mocks, mocker
):
    processor, _, _ = document_processor_with_mocks

    mock_faiss_from_texts = mocker.patch("app.core.document_processor.FAISS.from_texts")

    empty_chunks = []

    with pytest.raises(
        VectorStoreError, match="Cannot create vector store from empty chunks."
    ):
        processor.create_vector_store(empty_chunks)

    mock_faiss_from_texts.assert_not_called()


def test_document_processor_create_vector_store_raises_error_on_faiss_failure(
    document_processor_with_mocks, mocker
):
    processor, mock_embedding_model, _ = document_processor_with_mocks

    faiss_error = RuntimeError("Simulated FAISS creation error")
    mock_faiss_from_texts = mocker.patch(
        "app.core.document_processor.FAISS.from_texts",
        side_effect=faiss_error,
    )

    input_chunks = ["chunk A", "chunk B"]

    with pytest.raises(VectorStoreError) as excinfo:
        processor.create_vector_store(input_chunks)

    assert isinstance(excinfo.value, VectorStoreError)

    assert "Error during FAISS vector store creation" in str(excinfo.value)

    assert excinfo.value.__cause__ is faiss_error

    mock_faiss_from_texts.assert_called_once_with(
        texts=input_chunks, embedding=mock_embedding_model.model
    )
