from langchain_core.language_models.base import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.exceptions import LLMServiceError


def get_chat_llm(model_name: str, api_key: str) -> BaseLanguageModel:
    """
    Initializes the Google Gemini LLM model.
    Args:
        model_name (str): The name of the model to be used.
        api_key (str): The API key for authentication.
    Returns:
        BaseLanguageModel: An instance of the specified LLM model.
    Raises:
        LLMServiceError: If the model initialization fails.
    """
    try:
        if model_name == "gemini-2.0-flash-lite":
            llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        elif model_name == "gpt-4.1-mini":
            llm = ChatOpenAI(model=model_name, api_key=api_key)
        return llm
    except Exception as e:
        raise LLMServiceError(
            f"Failed to initialize the Google Gemini LLM model: {model_name}. Error: {e}"
        ) from e
