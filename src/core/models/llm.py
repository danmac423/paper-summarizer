from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.core.exceptions import LLMServiceError


def get_chat_llm(model_name: str, api_key: str) -> BaseChatModel:
    """
    Initializes the Google Gemini LLM model.
    Args:
        model_name (str): The name of the model to be used.
        api_key (str): The API key for authentication.
    Returns:
        BaseChatModel: An instance of the specified LLM model.
    Raises:
        LLMServiceError: If the model initialization fails.
    """
    try:
        if model_name == "gemini-2.0-flash-lite":
            llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
        elif model_name == "gpt-4.1-mini":
            llm = ChatOpenAI(model=model_name, api_key=api_key)
        else:
            raise LLMServiceError(
                f"Unsupported model name: {model_name}. Supported models are: gemini-2.0-flash-lite, gpt-4.1-mini."
            )
        return llm
    except LLMServiceError as e:
        raise e
    except Exception as e:
        raise LLMServiceError(
            f"Failed to initialize the LLM model: {model_name}."
        ) from e
