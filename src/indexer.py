from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings
from src.transcript import (
    TranscriptSegment,
    fetch_transcript,
    format_timestamp,
    youtube_watch_url,
)


@dataclass(frozen=True)
class VideoIndex:
    vector_store: FAISS
    chunk_count: int
    segment_count: int


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def create_transcript_documents(
    video_id: str,
    segments: Sequence[TranscriptSegment],
) -> list[Document]:
    """Split caption text and retain the timestamp span for each chunk."""
    usable_segments = [segment for segment in segments if segment.text.strip()]
    if not usable_segments:
        raise ValueError("The transcript is empty.")

    text_parts: list[str] = []
    segment_offsets: list[tuple[int, int, TranscriptSegment]] = []
    offset = 0
    for segment in usable_segments:
        text_parts.append(segment.text)
        start_offset = offset
        offset += len(segment.text)
        segment_offsets.append((start_offset, offset, segment))
        text_parts.append(" ")
        offset += 1
    transcript_text = "".join(text_parts).strip()

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(transcript_text)
    segment_starts = [entry[0] for entry in segment_offsets]

    documents: list[Document] = []
    search_from = 0
    for chunk in chunks:
        chunk_start = transcript_text.find(chunk, max(0, search_from))
        if chunk_start < 0:
            chunk_start = max(0, search_from)
        chunk_end = min(len(transcript_text), chunk_start + len(chunk))
        first_index = max(0, bisect_right(segment_starts, chunk_start) - 1)
        last_index = max(0, bisect_right(segment_starts, chunk_end) - 1)
        first_segment = segment_offsets[first_index][2]
        last_segment = segment_offsets[min(last_index, len(segment_offsets) - 1)][2]
        start_seconds = int(first_segment.start)
        end_seconds = int(last_segment.start + last_segment.duration)
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "start_seconds": start_seconds,
                    "end_seconds": max(start_seconds, end_seconds),
                    "timestamp": format_timestamp(start_seconds),
                    "source_url": youtube_watch_url(video_id, start_seconds),
                },
            )
        )
        search_from = chunk_start + 1
    return documents


def build_video_index(
    video_id: str,
    progress_callback: Callable[[str], None] | None = None,
) -> VideoIndex:
    def report(message: str) -> None:
        if progress_callback:
            progress_callback(message)

    report("Fetching captions")
    segments = fetch_transcript(video_id)
    report("Preparing timestamped chunks")
    documents = create_transcript_documents(video_id, segments)
    report("Loading embedding model")
    embeddings = get_embeddings()
    report("Building FAISS search index")
    vector_store = FAISS.from_documents(documents, embeddings)
    return VideoIndex(
        vector_store=vector_store,
        chunk_count=len(documents),
        segment_count=len(segments),
    )