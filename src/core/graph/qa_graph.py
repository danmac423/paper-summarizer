from typing import Optional

from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict

from src.config import K_RETRIEVED_DOCS


class GraphState(TypedDict):
    """State for the Q&A graph."""

    question: str
    context: List[Document]
    answer: str
    chat_history: Optional[List[BaseMessage]]


def retrieve(
    state: GraphState, vector_store: VectorStore, k_retrieved_docs: int = 3
) -> GraphState:
    """
    Retrieve relevant documents from the vector store based on the question.
    Args:
        state (GraphState): The current state of the graph.
        vector_store (VectorStore): The vector store to search for documents.
        k_retrieved_docs (int): The number of documents to retrieve.
    Returns:
        GraphState: The updated state with the retrieved documents.
    """
    question = state["question"]
    retrieved_docs = vector_store.similarity_search(question, k=k_retrieved_docs)
    return GraphState(question=question, context=retrieved_docs)


def generate(state: GraphState, llm: BaseChatModel) -> GraphState:
    """
    Generate an answer based on the retrieved documents and the question.
    Args:
        state (GraphState): The current state of the graph.
        llm (BaseChatModel): The language model to generate the answer.
    Returns:
        GraphState: The updated state with the generated answer.
    """
    question = state["question"]
    context = state["context"]
    chat_history = state.get("chat_history", [])

    docs_content = "\n\n".join(doc.page_content for doc in context)

    rag_prompt_messages = [
        (
            "system",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the user's question:
Context: {context}

If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise.
Consider the previous conversation history to provide a conversational and context-aware response.
Avoid repeating information if it was already discussed in the chat history, unless the user explicitly asks for clarification or repetition.
Base your answer primarily on the provided context for the current question.""",
        ),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
    ]

    rag_prompt = ChatPromptTemplate.from_messages(rag_prompt_messages)
    prompt_input = {"question": question, "context": docs_content}
    if chat_history:
        prompt_input["chat_history"] = chat_history

    messages = rag_prompt.invoke(prompt_input)

    response = llm.invoke(messages)

    return GraphState(
        question=question,
        context=context,
        answer=response.content,
        chat_history=chat_history,
    )


def build_qa_graph(vector_store: VectorStore, llm: BaseChatModel) -> CompiledStateGraph:
    """
    Build the Q&A graph using the provided vector store and language model.
    Args:
        vector_store (VectorStore): The vector store for document retrieval.
        llm (BaseChatModel): The language model for answer generation.
    Returns:
        CompiledStateGraph: The compiled state graph for the Q&A process.
    """
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node(
        "retrieve", lambda state: retrieve(state, vector_store, K_RETRIEVED_DOCS)
    )
    graph_builder.add_node("generate", lambda state: generate(state, llm))

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    return graph
