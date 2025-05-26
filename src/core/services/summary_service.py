from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.core.exceptions import SummaryServiceError


def generate_summary(full_text: str, llm: BaseChatModel, n_words: int = 100) -> str:
    """
    Generates a summary of the provided text using the specified LLM model.
    Args:
        full_text (str): The text to be summarized.
        llm (BaseChatModel): The LLM model to be used for summarization.
        n_words (int): The number of words for the final summary.
    Returns:
        str: The generated summary.
    Raises:
        SummaryServiceError: If there is an error during the summary generation process.
    """
    try:
        combine_prompt_template = """
        Given the following summaries of a scientific article:
        "{text}"

        Please combine them into a single, coherent summary of the entire article.
        The final summary should be **around {n_words} words long** and highlight the main findings and contributions.
        Final Summary:
        """

        combine_chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a highly skilled assistant specializing in summarizing complex scientific articles into concise and coherent summaries.",
                ),
                ("user", combine_prompt_template),
            ]
        )

        summary_chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            combine_prompt=combine_chat_prompt,
        )

        docs = [Document(page_content=full_text)]

        summary = summary_chain.invoke(
            input={"input_documents": docs, "n_words": n_words}
        )

        return summary.get("output_text")

    except Exception as e:
        raise SummaryServiceError(f"Error during summary generation: {e}") from e
