import streamlit as st
from app.web_ui.state import update_session_state_on_input_change
from app.web_ui.processing import process_uploaded_file


def render_sidebar():
    """Renders the sidebar elements and returns the user inputs."""
    st.sidebar.header("Configuration")
    llm_model_name = st.sidebar.selectbox(
        "Choose LLM model", ["gemini-2.0-flash-lite", "gpt-4.1-mini"]
    )
    api_key = st.sidebar.text_input(
        "Your API Key",
        key="api_key",
        type="password",
    )
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF file", type=["pdf"], label_visibility="collapsed"
    )

    update_session_state_on_input_change(uploaded_file, llm_model_name, api_key)

    if st.session_state.processed_article:
        st.sidebar.success(
            f"File '{st.session_state.processed_article['name']}' processed successfully"
        )

    can_process = (
        uploaded_file is not None and st.session_state.processed_article is None
    )
    if can_process:
        if st.sidebar.button("Process Paper", use_container_width=True, type="primary"):
            with st.spinner(
                f"Processing '{uploaded_file.name}'... It may take a while."
            ):
                process_uploaded_file(uploaded_file)
            st.rerun()


def render_intro():
    """Renders the introductory text on the main page."""
    st.markdown("---")
    st.markdown(
        """
        ## Welcome to the Paper Analyzer!

        This app helps you analyze and extract insights from scientific papers with ease.
        Simply upload a paper, and let the app process it to provide you with key information,
        summaries, and more.

        ### Features:
        - **Paper Summarization**: Get concise summaries of your papers.
        - **Q&A**: Ask questions about the paper and get instant answers.

        ### Supported Models:
        - **Gemini 2.0 Flash Lite**: A lightweight model for efficient processing.
        
        ### How to Use:
        1. Upload a PDF file of the paper you want to analyze.
        2. Choose the LLM model and enter your API key in the sidebar.
        3. Click on "Process Paper" to start the analysis.
        4. Navigate through the tabs to view summaries and ask questions about the paper.
        """
    )
