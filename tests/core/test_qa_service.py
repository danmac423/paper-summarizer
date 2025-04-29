import pytest

from app.core.qa_service import (
    BaseChatModel,
    QAServiceError,
    VectorStore,
    generate_qa_answer,
)


@pytest.fixture
def mock_dependencies(mocker):
    mock_vec = mocker.Mock(spec=VectorStore)
    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_question = "What is the capital of France?"

    mock_qa_graph = mocker.Mock()
    mock_qa_graph.invoke.return_value = {"answer": "Paris"}
    mocker.patch("app.core.qa_service.build_qa_graph", return_value=mock_qa_graph)
    return mock_vec, mock_llm, mock_question, mock_qa_graph


def test_generate_qa_answer_success(mock_dependencies):
    mock_vec, mock_llm, mock_question, mock_qa_graph = mock_dependencies

    answer = generate_qa_answer(mock_vec, mock_llm, mock_question)

    assert answer == "Paris"
    mock_qa_graph.invoke.assert_called_once_with({"question": mock_question})

def test_generate_qa_answer_failure(mocker, mock_dependencies):
    mock_vec, mock_llm, mock_question, mock_qa_graph = mock_dependencies
    mock_qa_graph.invoke.side_effect = Exception("Some error")

    with pytest.raises(QAServiceError) as excinfo:
        generate_qa_answer(mock_vec, mock_llm, mock_question)

    assert str(excinfo.value) == "Error during Q&A generation: Some error"
    mock_qa_graph.invoke.assert_called_once_with({"question": mock_question})