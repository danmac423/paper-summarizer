import pytest

from app.core.llm_service import (
    ChatGoogleGenerativeAI,
    ChatOpenAI,
    LLMServiceError,
    get_chat_llm,
)


@pytest.mark.parametrize(
    "model_name,expected_class",
    [
        ("gemini-2.0-flash-lite", ChatGoogleGenerativeAI),
        ("gpt-4.1-mini", ChatOpenAI),
    ],
)
def test_get_chat_llm_returns_instance_of_correct_class(
    mocker, model_name, expected_class
):
    mock_api_key = "test_api_key"
    mock_llm_instance = mocker.Mock(spec=expected_class)
    mock_llm_class = mocker.patch(
        f"app.core.llm_service.{expected_class.__name__}",
        return_value=mock_llm_instance,
    )
    llm = get_chat_llm(model_name, mock_api_key)
    mock_llm_class.assert_called_once_with(model=model_name, api_key=mock_api_key)
    assert isinstance(llm, expected_class)


def test_get_chat_llm_raises_error_on_unsupported_model():
    api_key = "test_api_key"
    model_name = "unsupported-model"

    with pytest.raises(LLMServiceError) as excinfo:
        get_chat_llm(model_name, api_key)

    assert str(excinfo.value) == (
        "Unsupported model name: unsupported-model. Supported models are: gemini-2.0-flash-lite, gpt-4.1-mini."
    )


@pytest.mark.parametrize(
    "model_name,expected_class",
    [
        ("gemini-2.0-flash-lite", ChatGoogleGenerativeAI),
        ("gpt-4.1-mini", ChatOpenAI),
    ],
)
def test_get_chat_llm_raises_error_on_initialization_failure(
    mocker, model_name, expected_class
):
    mock_api_key = "test_api_key"
    mocker.patch(
        f"app.core.llm_service.{expected_class.__name__}",
        side_effect=Exception("Simulated initialization error"),
    )

    with pytest.raises(LLMServiceError) as excinfo:
        get_chat_llm(model_name, mock_api_key)

    assert str(excinfo.value) == (f"Failed to initialize the LLM model: {model_name}.")
