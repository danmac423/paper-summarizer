from langchain_core.messages import HumanMessage

from src.core.graph.qa_graph import (
    BaseChatModel,
    CompiledStateGraph,
    Document,
    GraphState,
    VectorStore,
    build_qa_graph,
    generate,
    retrieve,
)


def test_retrieve(mocker):
    mock_vector_store = mocker.Mock()
    mock_vector_store.similarity_search.return_value = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]

    state = {
        "question": "Test question",
        "context": [],
        "answer": None,
        "chat_history": [HumanMessage(content="Previous user message")],
    }

    updated_state = retrieve(state, mock_vector_store, k_retrieved_docs=3)

    assert updated_state["context"] == [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]


def test_generate_with_history(mocker):
    test_question = "Test question"
    test_docs = [
        mocker.Mock(spec=Document, page_content="doc1"),
        mocker.Mock(spec=Document, page_content="doc2"),
        mocker.Mock(spec=Document, page_content="doc3"),
    ]
    initial_state: GraphState = {
        "question": test_question,
        "context": test_docs,
        "answer": None,
        "chat_history": [HumanMessage(content="Previous chat message")],
    }
    expected_answer = "Generated answer"

    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_llm_response = mocker.Mock()
    mock_llm_response.content = expected_answer
    mock_llm.invoke.return_value = mock_llm_response

    updated_state = generate(initial_state, mock_llm)

    assert updated_state.get("question") == test_question
    assert updated_state.get("context") == test_docs
    assert updated_state.get("answer") == expected_answer
    assert updated_state.get("chat_history") == [HumanMessage(content="Previous chat message")]


def test_generate_without_history(mocker):
    test_question = "Test question"
    test_docs = [
        mocker.Mock(spec=Document, page_content="doc1"),
        mocker.Mock(spec=Document, page_content="doc2"),
        mocker.Mock(spec=Document, page_content="doc3"),
    ]
    initial_state: GraphState = {
        "question": test_question,
        "context": test_docs,
        "answer": None,
    }
    expected_answer = "Generated answer"

    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_llm_response = mocker.Mock()
    mock_llm_response.content = expected_answer
    mock_llm.invoke.return_value = mock_llm_response

    updated_state = generate(initial_state, mock_llm)

    assert updated_state.get("question") == test_question
    assert updated_state.get("context") == test_docs
    assert updated_state.get("answer") == expected_answer
    assert updated_state.get("chat_history") == []


def test_build_qa_graph(mocker):
    mock_vector_store = mocker.Mock(spec=VectorStore)
    mock_llm = mocker.Mock(spec=BaseChatModel)

    graph = build_qa_graph(mock_vector_store, mock_llm)

    assert isinstance(graph, CompiledStateGraph)
    assert len(graph.nodes) == 3

    assert "__start__" in graph.nodes
    assert "retrieve" in graph.nodes
    assert "generate" in graph.nodes
