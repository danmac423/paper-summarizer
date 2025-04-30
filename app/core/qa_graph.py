from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import List, TypedDict

from config.config import K_RETRIEVED_DOCS


class GraphState(TypedDict):
    """State for the Q&A graph."""

    question: str
    context: List[Document]
    answer: str


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

    docs_content = "\n\n".join(doc.page_content for doc in context)

    # rag_prompt = hub.pull("rlm/rag-prompt")

    rag_prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. Use ten sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)

    messages = rag_prompt.invoke({"question": question, "context": docs_content})
    response = llm.invoke(messages)

    return GraphState(
        question=question,
        context=context,
        answer=response.content,
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
    graph_builder.add_node("generate", lambda state: generate(state, llm))  # Pass llm

    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()

    return graph
