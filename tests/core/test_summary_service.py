import pytest

from app.core.summary_service import (
    BaseChatModel,
    Document,
    SummaryServiceError,
    generate_summary,
)


def test_generate_summary_success(mocker):
    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_chain = mocker.Mock()
    mock_chain.invoke.return_value = {"output_text": "This is a summary."}
    mocker.patch(
        "app.core.summary_service.load_summarize_chain", return_value=mock_chain
    )

    full_text = "This is a long text that needs to be summarized."
    n_words = 100
    expected_summary = "This is a summary."
    summary = generate_summary(full_text, mock_llm, n_words)
    assert summary == expected_summary
    mock_chain.invoke.assert_called_once_with(
        input={
            "input_documents": [Document(page_content=full_text)],
            "n_words": n_words,
        }
    )


def test_generate_summary_failure(mocker):
    mock_llm = mocker.Mock(spec=BaseChatModel)
    mock_chain = mocker.Mock()
    mock_chain.invoke.side_effect = Exception("Some error")
    mocker.patch(
        "app.core.summary_service.load_summarize_chain", return_value=mock_chain
    )

    full_text = "This is a long text that needs to be summarized."
    n_words = 100

    with pytest.raises(SummaryServiceError) as excinfo:
        generate_summary(full_text, mock_llm, n_words)

    assert str(excinfo.value) == "Error during summary generation: Some error"
    mock_chain.invoke.assert_called_once_with(
        input={
            "input_documents": [Document(page_content=full_text)],
            "n_words": n_words,
        }
    )
