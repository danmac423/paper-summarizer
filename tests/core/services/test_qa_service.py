import pytest
from langchain_core.messages import BaseMessage  # Ensure BaseMessage is imported

from src.core.services.qa_service import (
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

    mock_history_messages = [mocker.Mock(spec=BaseMessage), mocker.Mock(spec=BaseMessage)]

    mock_qa_graph = mocker.Mock()

    def invoke_side_effect(input_dict):
        if input_dict["chat_history"] is None:
            return {"answer": "Paris"}
        elif input_dict["chat_history"] == mock_history_messages:
            return {"answer": "Paris, as we discussed."}
        return {"answer": "Default answer"}

    mock_qa_graph.invoke.side_effect = invoke_side_effect
    mocker.patch("src.core.services.qa_service.build_qa_graph", return_value=mock_qa_graph)
    return mock_vec, mock_llm, mock_question, mock_qa_graph, mock_history_messages


def test_generate_qa_answer_success_no_history(mock_dependencies):
    mock_vec, mock_llm, mock_question, mock_qa_graph, _ = mock_dependencies

    answer = generate_qa_answer(mock_vec, mock_llm, mock_question, history=None)

    assert answer == "Paris"
    mock_qa_graph.invoke.assert_called_once_with({"question": mock_question, "chat_history": None})


def test_generate_qa_answer_success_with_history(mock_dependencies):
    mock_vec, mock_llm, mock_question, mock_qa_graph, mock_history_messages = mock_dependencies

    answer = generate_qa_answer(mock_vec, mock_llm, mock_question, history=mock_history_messages)

    assert answer == "Paris, as we discussed."
    mock_qa_graph.invoke.assert_called_once_with({"question": mock_question, "chat_history": mock_history_messages})


def test_generate_qa_answer_failure(mocker, mock_dependencies):
    mock_vec, mock_llm, mock_question, mock_qa_graph, _ = mock_dependencies

    error_message = "Some error during graph execution"
    mock_qa_graph.invoke.side_effect = Exception(error_message)

    with pytest.raises(QAServiceError) as excinfo:
        generate_qa_answer(mock_vec, mock_llm, mock_question, history=None)

    assert str(excinfo.value) == f"Error during Q&A generation: {error_message}"
    mock_qa_graph.invoke.assert_called_once_with({"question": mock_question, "chat_history": None})
