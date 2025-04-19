from app.core.document_processor import split_text_into_chunks

import pytest


@pytest.mark.parametrize(
    "text, chunk_size, chunk_overlap, expected_chunks",
    [
        (
            "This is a test text that needs to be split into smaller chunks.",
            20,
            5,
            [
                "This is a test text",
                "text that needs to",
                "to be split into",
                "into smaller",
                "chunks.",
            ],
        ),
        (
            "Another example of text splitting.",
            10,
            2,
            [
                "Another",
                "example",
                "of text",
                "splitting",
                "ng.",
            ],
        ),
    ],
)
def test_split_text_into_chunks(text, chunk_size, chunk_overlap, expected_chunks):
    """
    Test the split_text_into_chunks function.
    """
    chunks = split_text_into_chunks(text, chunk_size, chunk_overlap)
    assert chunks == expected_chunks
