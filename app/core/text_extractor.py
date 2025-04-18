import io
import pymupdf4llm
from pymupdf import Document


class TextExtractionError(Exception):
    """Custom exception for text extraction errors."""
    pass


def extract_text_from_pdf(content: bytes) -> str:
    """
    Extract text from a PDF file and convert it to Markdown format.
    Args:
        content (bytes): The PDF file content as bytes.
    Returns:
        str: The extracted text in Markdown format.
    Raises:
        TextExtractionError: If there is an error during text extraction.
    """
    if not content:
        raise TextExtractionError("No content provided for extraction.")
    try:
        stream = io.BytesIO(content)
        pdf = Document(stream=stream)
        md_text = pymupdf4llm.to_markdown(pdf)
        return md_text
    except Exception as e:
        raise TextExtractionError(f"Text extraction error: {e}")

