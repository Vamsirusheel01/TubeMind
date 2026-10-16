from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import get_settings
from src.transcript import format_timestamp


def create_llm(api_key: str, model_name: str | None = None) -> ChatGoogleGenerativeAI:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name or settings.chat_model,
        temperature=0.2,
        google_api_key=api_key,
    )


def answer_question(
    question: str,
    retriever,
    llm: ChatGoogleGenerativeAI,
    history: Sequence[dict[str, str]] = (),
) -> tuple[str, list[Document]]:
    """Retrieve relevant caption chunks and answer using transcript evidence."""
    documents = retriever.invoke(question)
    if not documents:
        return "I couldn't find relevant captions for that question.", []

    context = "\n\n".join(
        f"[{format_timestamp(document.metadata.get('start_seconds', 0))}] "
        f"{document.page_content}"
        for document in documents
    )
    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}"
        for message in history[-6:]
        if message.get("role") in {"user", "assistant"}
    )
    prompt = PromptTemplate(
        template=(
            "You are TubeMind, a helpful assistant for YouTube videos.\n"
            "Answer using only the transcript excerpts. Use conversation history "
            "to understand follow-up questions, but do not treat it as evidence. "
            "If the excerpts do not support an answer, say you don't know based "
            "on the available transcript. Do not invent facts.\n\n"
            "Conversation history:\n{history}\n\n"
            "Transcript excerpts:\n{context}\n\n"
            "Question: {question}\n"
            "Answer clearly and concisely."
        ),
        input_variables=["context", "history", "question"],
    )
    answer = (prompt | llm | StrOutputParser()).invoke(
        {
            "context": context,
            "history": conversation or "No previous messages.",
            "question": question,
        }
    )
    return answer, documents