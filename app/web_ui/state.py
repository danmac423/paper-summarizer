import streamlit as st


def initialize_session_state():
    """Initializes necessary variables in Streamlit's session state."""
    if "processed_article" not in st.session_state:
        st.session_state.processed_article = None
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None
    if "summary_text" not in st.session_state:
        st.session_state.summary_text = None
    if "summary_error" not in st.session_state:
        st.session_state.summary_error = None
    if "processing_error" not in st.session_state:
        st.session_state.processing_error = None
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "llm_model_name": None,
            "api_key": None,
        }


def update_session_state_on_input_change(uploaded_file, llm_model_name, api_key):
    """
    Resets parts of the session state if the uploaded file or LLM config changes.
    """
    current_uploaded_filename = uploaded_file.name if uploaded_file else None

    # Reset state if a new file is uploaded
    if current_uploaded_filename != st.session_state.last_uploaded_filename:
        st.session_state.processed_article = None
        st.session_state.processing_error = None
        st.session_state.last_uploaded_filename = current_uploaded_filename
        st.session_state.summary_text = None
        st.session_state.summary_error = None
    
    if (
        st.session_state.llm_config["llm_model_name"] != llm_model_name
        or st.session_state.llm_config["api_key"] != api_key
    ):
        st.session_state.llm_config["llm_model_name"] = llm_model_name
        st.session_state.llm_config["api_key"] = api_key
