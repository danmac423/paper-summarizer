from langchain_google_genai import ChatGoogleGenerativeAI
import os

from app.core.exceptions import LLMServiceError


def get_chat_llm(model_name: str) -> ChatGoogleGenerativeAI:
    """
    Initializes the Google Gemini LLM model.
    Args:
        model_name (str): The name of the model to be used.
    Returns:
        ChatGoogleGenerativeAI: An instance of the Google Gemini LLM model.
    """
    if not os.getenv("GOOGLE_API_KEY"):
        raise LLMServiceError("GOOGLE_API_KEY environment variable not set.")
    try:
        llm = ChatGoogleGenerativeAI(model=model_name)
        return llm
    except Exception as e:
        raise LLMServiceError(
            f"Failed to initialize the Google Gemini LLM model: {model_name}. Error: {e}"
        ) from e
