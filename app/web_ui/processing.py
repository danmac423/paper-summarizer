import os
import tempfile
from io import BytesIO

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.document_processor import DocumentProcessor
from app.core.embedding_model import EmbeddingModel
from app.core.text_extractor import MarkerTextExtractor
from config.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEVICE,
    EMBEDDING_MODEL_NAME,
    SEPARATORS,
)


def process_uploaded_file(uploaded_file: BytesIO):
    """
    Processes the uploaded PDF file, extracts text, splits it into chunks,
    and creates a vector store for further processing.
    Args:
        uploaded_file (BytesIO): The uploaded PDF file.
    """
    if uploaded_file is None:
        st.session_state.processing_error = "No file uploaded."
        st.session_state.processed_article = None
        return

    tmp_file_path = None
    try:
        st.session_state.summary_text = None
        st.session_state.summary_error = None

        file_contents = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_contents)
            tmp_file_path = tmp_file.name

        text_extractor = MarkerTextExtractor()
        article_text = text_extractor.extract_text_from_pdf_file(tmp_file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=SEPARATORS,
        )
        embedding_model = EmbeddingModel(device=DEVICE, model_name=EMBEDDING_MODEL_NAME)

        doc_processor = DocumentProcessor(
            embedding_model=embedding_model, text_splitter=text_splitter
        )

        chunks = doc_processor.split_text(article_text)
        vector_store = doc_processor.create_vector_store(chunks)

        st.session_state.processed_article = {
            "name": uploaded_file.name,
            "text": article_text,
            "vector_store": vector_store,
            "chunks": chunks,
        }
        st.session_state.processing_error = None

    except Exception as e:
        st.session_state.processing_error = str(e)
        st.session_state.processed_article = None

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
