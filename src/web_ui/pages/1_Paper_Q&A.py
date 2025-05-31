import streamlit as st
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.messages import trim_messages

from src.config import HISTORY_MAX_LENGTH
from src.core.models.llm import get_chat_llm
from src.core.services.qa_service import generate_qa_answer
from src.web_ui.state import initialize_session_state
from src.web_ui.ui import render_intro, render_sidebar

initialize_session_state()

st.set_page_config(page_title="Paper Q&A", layout="wide")
st.title("📝 Paper Q&A")

render_sidebar()

if st.session_state.processing_error:
    st.error(st.session_state.processing_error)

if not st.session_state.processed_article:
    render_intro()

if st.session_state.processed_article:
    api_key = st.session_state.llm_config.get("api_key")
    llm_model_name = st.session_state.llm_config.get("llm_model_name")
    processed_article = st.session_state.processed_article

    chat_history: StreamlitChatMessageHistory = st.session_state.chat_history

    for message in chat_history.messages:
        st.chat_message(message.type).write(message.content)

    if not api_key:
        st.error("API Key is missing. Please provide it in the sidebar to ask about the paper.")

    can_prompt = api_key and processed_article is not None and processed_article.get("vector_store") is not None

    if prompt := st.chat_input(placeholder="Ask a question about the paper", disabled=not can_prompt):
        st.chat_message("user").write(prompt)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_chat_llm(model_name=llm_model_name, api_key=api_key)

                    messages = trim_messages(
                        chat_history.messages, strategy="last", token_counter=len, max_tokens=HISTORY_MAX_LENGTH
                    )

                    answer = generate_qa_answer(
                        vec=processed_article["vector_store"],
                        llm=llm,
                        question=prompt,
                        history=messages,
                    )

                    chat_history.add_user_message(prompt)
                    chat_history.add_ai_message(answer)
                    st.write(answer)
                except Exception as e:
                    answer = "Error occured while generating the answer."
                    chat_history.add_ai_message(answer)
                    st.write(answer)
                    st.error(e)
