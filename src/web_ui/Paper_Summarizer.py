import streamlit as st

from src.core.models.llm import get_chat_llm
from src.core.services.summary_service import generate_summary
from src.web_ui.state import initialize_session_state
from src.web_ui.ui import render_intro, render_sidebar


initialize_session_state()

st.set_page_config(page_title="Paper Summarizer", layout="wide")
st.title("📝 Paper Summarizer")

render_sidebar()

if st.session_state.processing_error:
    st.error(st.session_state.processing_error)

if not st.session_state.processed_article:
    render_intro()


if st.session_state.processed_article:
    api_key = st.session_state.llm_config.get("api_key")
    llm_model_name = st.session_state.llm_config.get("llm_model_name")
    processed_article = st.session_state.processed_article

    summary_n_words = st.number_input(
        "Desired words in summary",
        value=100,
        format="%d",
        step=10,
        help="Enter the desired number of words for the summary.",
    )

    if not api_key:
        st.error(
            "API Key is missing. Please provide it in the sidebar to generate a summary."
        )

    if summary_n_words <= 0:
        st.error("Please enter a positive number for the desired words in summary.")

    can_generate_summary = (
        api_key
        and processed_article is not None
        and processed_article.get("text") is not None
        and summary_n_words > 0
    )

    if can_generate_summary:
        col1, col2 = st.columns([3, 1])
        with col2:
            generate_summary_button = st.button(
                "Generate Summary",
                type="primary",
                key="generate_summary",
                use_container_width=True,
            )

        if generate_summary_button:
            st.session_state.summary_text = None
            st.session_state.summary_error = None
            with st.spinner("Generating summary..."):
                try:
                    llm_instance = get_chat_llm(
                        model_name=llm_model_name, api_key=api_key
                    )

                    summary = generate_summary(
                        processed_article["text"],
                        llm_instance,
                        n_words=summary_n_words,
                    )

                    st.session_state.summary_text = summary
                    st.session_state.summary_error = None

                except Exception as e:
                    st.session_state.summary_error = str(e)
                    st.session_state.summary_text = None

    if st.session_state.summary_error:
        st.error(st.session_state.summary_error)

    if st.session_state.summary_text:
        st.subheader("Article Summary:")
        st.write(st.session_state.summary_text)
