from app.core.qa_graph import (
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
    }

    updated_state = retrieve(state, mock_vector_store, k_retrieved_docs=3)

    assert updated_state["context"] == [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]


def test_generate(mocker):
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
    expected_docs_content = "doc1\n\ndoc2\n\ndoc3"
    expected_answer = "Generated answer"

    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_llm_response = mocker.Mock()
    mock_llm_response.content = expected_answer
    mock_llm.invoke.return_value = mock_llm_response

    mock_rag_prompt = mocker.Mock()
    mock_prompt_messages = ["Test message"]
    mock_rag_prompt.invoke.return_value = mock_prompt_messages

    mock_hub_pull = mocker.patch(
        "app.core.qa_graph.hub.pull", return_value=mock_rag_prompt
    )

    updated_state = generate(initial_state, mock_llm)

    mock_hub_pull.assert_called_once_with("rlm/rag-prompt")
    mock_rag_prompt.invoke.assert_called_once_with(
        {"question": test_question, "context": expected_docs_content}
    )
    mock_llm.invoke.assert_called_once_with(mock_prompt_messages)
    assert updated_state.get("question") == test_question
    assert updated_state.get("context") == test_docs
    assert updated_state.get("answer") == expected_answer


def test_build_qa_graph(mocker):
    mock_vector_store = mocker.Mock(spec=VectorStore)
    mock_llm = mocker.Mock(spec=BaseChatModel)

    graph = build_qa_graph(mock_vector_store, mock_llm)

    assert isinstance(graph, CompiledStateGraph)
    assert len(graph.nodes) == 3

    assert "__start__" in graph.nodes
    assert "retrieve" in graph.nodes
    assert "generate" in graph.nodes
