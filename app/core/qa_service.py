from langchain.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.exceptions import QAServiceError
from app.core.qa_graph import build_qa_graph


def generate_qa_answer(vec: VectorStore, llm: BaseChatModel, question: str) -> str:
    """
    Generates an answer to a question using the provided vector store and language model.
    Args:
        vec (VectorStore): The vector store to search for relevant documents.
        llm (BaseChatModel): The language model to generate the answer.
        question (str): The question to be answered.
    Returns:
        str: The generated answer.
    Raises:
        QAServiceError: If there is an error during the Q&A generation process.
    """
    try:
        qa_graph = build_qa_graph(vector_store=vec, llm=llm)
        response = qa_graph.invoke({"question": question})
        answer = response["answer"]
        return answer
    except Exception as e:
        raise QAServiceError(f"Error during Q&A generation: {e}") from e
