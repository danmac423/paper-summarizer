import streamlit as st
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import components from your core library
from app.core.text_extractor import MarkerTextExtractor
from app.core.document_processor import DocumentProcessor
from app.core.embedding_model import EmbeddingModel


def process_uploaded_file(uploaded_file, device):
    """
    Handles the end-to-end processing of the uploaded PDF file:
    extraction, embedding, chunking, and vector store creation.
    Updates session state with results or errors.
    """
    if uploaded_file is None:
        st.session_state.processing_error = "No file uploaded."
        st.session_state.processed_article = None
        return  # Exit if no file

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

        # Define text splitter and embedding model configuration
        separators = ["\n\n", "\n", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=separators,
        )
        embedding_model = EmbeddingModel(device=device)  # Use passed device

        doc_processor = DocumentProcessor(
            embedding_model=embedding_model, text_splitter=text_splitter
        )

        chunks = doc_processor.split_text(article_text)
        vector_store = doc_processor.create_vector_store(chunks)

        # Store processed results in session state
        st.session_state.processed_article = {
            "name": uploaded_file.name,
            "text": article_text,
            "vector_store": vector_store,
            "chunks": chunks,
        }
        st.session_state.processing_error = None  # Clear any previous error

    except Exception as e:
        # Catch and store processing errors
        st.session_state.processing_error = f"Error during processing: {str(e)}"
        st.session_state.processed_article = None  # Clear processed article on error

    finally:
        # Ensure temporary file is cleaned up
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
