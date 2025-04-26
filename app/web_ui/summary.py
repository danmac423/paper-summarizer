import streamlit as st
import os

# Import components from your core library
from app.core.llm_service import get_chat_llm
from app.core.summary_service import generate_summary


def handle_summary_tab(tab):
    """
    Renders the content of the Summary tab and handles summary generation logic.
    """
    with tab:
        st.header("Summary of the Article")
        summary_n_words = st.number_input(
            "Desired words in summary",
            value=100,
            format="%d",
            step=10,
            help="Enter the desired number of words for the summary.",
        )

        api_key = st.session_state.llm_config.get("api_key")
        llm_model_name = st.session_state.llm_config.get("llm_model_name")
        processed_article = st.session_state.processed_article

        can_generate_summary = (
            api_key
            and processed_article
            and processed_article.get("text")
            and summary_n_words > 0
        )

        if not api_key and processed_article:
            st.error(
                "API Key is missing. Please provide it in the sidebar to generate a summary."
            )

        if summary_n_words <= 0 and processed_article:
            st.error("Please enter a positive number for the desired words in summary.")

        # Display errors or summary from session state
        if st.session_state.summary_error:
            st.error(st.session_state.summary_error)

        if st.session_state.summary_text:
            st.subheader("Article Summary:")
            st.write(st.session_state.summary_text)

        # Render the Generate Summary button
        if can_generate_summary:
            if st.button("Generate Summary"):
                st.session_state.summary_text = None
                st.session_state.summary_error = None
                with st.spinner("Generating summary..."):
                    try:
                        # Set API key environment variable for the LLM service
                        os.environ["GOOGLE_API_KEY"] = api_key
                        llm_instance = get_chat_llm(llm_model_name)

                        # Generate summary using the core service
                        summary = generate_summary(
                            processed_article["text"],
                            llm_instance,
                            n_words=summary_n_words,
                        )

                        # Store result in session state
                        st.session_state.summary_text = summary
                        st.session_state.summary_error = None  # Clear error

                    except Exception as e:
                        # Catch and store summary generation errors
                        st.session_state.summary_error = (
                            f"Error generating summary: {str(e)}"
                        )
                        st.session_state.summary_text = None  # Clear summary on error

                    # Rerun needed to update the UI after state changes
                    st.rerun()
