from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    google_api_key: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_count: int


def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        chat_model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retrieval_count=int(os.getenv("RETRIEVAL_COUNT", "4")),
    )


def get_google_api_key(secret_key: str | None = None) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or secret_key
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY in your .env file or Streamlit deployment secrets."
        )
    return api_key