from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text_into_chunks(
    text: str, chunk_size: int = 200, chunk_overlap: int = 50
) -> list[str]:
    """
    Splits the text into chunks using RecursiveCharacterTextSplitter.
    Args:
        text (str): The text to be split.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
    Returns:
        list[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_text(text)
    return chunks
