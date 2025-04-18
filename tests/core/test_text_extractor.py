import pytest
from unittest.mock import patch, MagicMock
from app.core.text_extractor import extract_text_from_pdf, TextExtractionError


@patch("app.core.text_extractor.Document")
@patch("app.core.text_extractor.pymupdf4llm.to_markdown")
def test_empty_content(MockToMarkdown, MockDocumentInit):
    """
    Test the behavior when no content is provided for extraction.
    """
    with pytest.raises(
        TextExtractionError, match="No content provided for extraction."
    ):
        extract_text_from_pdf(b"")


@patch("app.core.text_extractor.Document")
@patch("app.core.text_extractor.pymupdf4llm.to_markdown")
def test_text_extraction_error(MockToMarkdown, MockDocumentInit):
    """
    Test the behavior when an error occurs during text extraction.
    """
    MockDocumentInit.side_effect = Exception("Some error")
    with pytest.raises(TextExtractionError, match="Text extraction error: Some error"):
        extract_text_from_pdf(b"valid content")
    MockDocumentInit.assert_called_once()
    MockToMarkdown.assert_not_called()


@patch("app.core.text_extractor.Document")
@patch("app.core.text_extractor.pymupdf4llm.to_markdown")
def test_successful_text_extraction(MockToMarkdown, MockDocumentInit):
    """
    Test the successful extraction of text from a PDF file.
    """
    mock_pdf = MagicMock()
    MockDocumentInit.return_value = mock_pdf
    MockToMarkdown.return_value = "extracted text"

    result = extract_text_from_pdf(b"valid content")

    MockDocumentInit.assert_called_once()
    MockToMarkdown.assert_called_once_with(mock_pdf)
    assert result == "extracted text"
