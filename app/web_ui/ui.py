import streamlit as st
from app.web_ui.summary import handle_summary_tab  # Import handler for summary tab


def render_sidebar():
    """Renders the sidebar elements and returns the user inputs."""
    st.sidebar.header("Configuration")
    llm_model_name = st.sidebar.selectbox(
        "Choose LLM model", ["gemini-2.0-flash-lite"], index=0
    )
    api_key = st.sidebar.text_input("Your API Key", type="password")
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF file", type=["pdf"], label_visibility="collapsed"
    )
    return llm_model_name, api_key, uploaded_file


def display_processing_error():
    """Displays the processing error if it exists in session state."""
    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)


def render_main_tabs():
    """Renders the main content area with tabs for Summary and Q&A."""
    tab_summary, tab_qa = st.tabs(["Summary", "Q&A (In Progress)"])

    # Delegate rendering and logic for each tab to specific handlers
    handle_summary_tab(tab_summary)

    with tab_qa:
        st.header("Q&A (In Progress)")
        st.info("This section is under development.")
        # Add Q&A interface elements here when implemented
