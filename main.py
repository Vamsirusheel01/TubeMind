"""Compatibility exports for the original single-file RAG prototype."""

from src.chain import answer_question, create_llm
from src.indexer import build_video_index
from src.transcript import extract_video_id, fetch_transcript

__all__ = [
    "answer_question",
    "build_video_index",
    "create_llm",
    "extract_video_id",
    "fetch_transcript",
]

if __name__ == "__main__":
    print("Run the portfolio application with: streamlit run app.py")