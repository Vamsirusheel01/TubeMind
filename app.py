
from __future__ import annotations

from typing import Any

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApiException,
)

from src.chain import answer_question, create_llm
from src.config import get_google_api_key, get_settings
from src.indexer import VideoIndex, build_video_index
from src.transcript import extract_video_id, youtube_watch_url

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="TubeMind | Video Intelligence",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="auto",
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#e9f0ed; --muted:#8e9a95; --canvas:#101513; --panel:#171e1b; --line:#29332e; --lime:#c5f277; --coral:#f18468; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--canvas); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:#151b18; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div { padding-top:1.6rem; }
.block-container { max-width:1440px; padding:2rem 3rem 5rem; }
.brand { display:flex; align-items:center; gap:.7rem; margin-bottom:.15rem; }
.brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:9px; background:var(--lime); color:#172014; font-size:15px; font-weight:700; }
.brand-name, .workspace-title { font-family:'Space Grotesk',sans-serif; font-weight:600; color:var(--ink); }
.brand-name { font-size:1.25rem; }
.workspace-title { font-size:1.12rem; margin:0 0 .2rem; }
.workspace-subtitle { color:var(--muted); font-size:.86rem; margin-bottom:1rem; }
.eyebrow { color:var(--lime); font:500 .72rem 'DM Mono',monospace; text-transform:uppercase; }
.hero { padding:.4rem 0 1.45rem; border-bottom:1px solid var(--line); margin-bottom:1.4rem; }
.hero h1 { font:600 clamp(1.7rem,3vw,2.45rem) 'Space Grotesk',sans-serif; margin:.35rem 0 .45rem; color:var(--ink); }
.hero p { color:var(--muted); font-size:.98rem; margin:0; }
.empty-panel { border:1px solid var(--line); border-radius:8px; padding:1.5rem; background:linear-gradient(145deg,#171e1b,#141a17); }
.empty-index { display:grid; place-items:center; width:48px; height:48px; border:1px solid #3b4941; border-radius:12px; color:var(--lime); font-size:1.25rem; margin-bottom:1rem; }
[data-testid="stChatMessage"] { border:1px solid var(--line); border-radius:8px; background:#171e1b; }
[data-testid="stChatMessage"] p { line-height:1.65; }
.stButton button { border-radius:6px; min-height:2.65rem; font-weight:600; }
.stButton button[kind="primary"] { background:var(--lime); border-color:var(--lime); color:#172014; }
.stTextInput input, [data-testid="stChatInput"] textarea { background:#171e1b; border-color:var(--line); border-radius:7px; color:var(--ink); }
[data-testid="stMetric"] { border:1px solid var(--line); border-radius:7px; padding:.75rem; background:#171e1b; }
[data-testid="stMetricLabel"] { color:var(--muted); }
a { color:var(--lime) !important; }
.sidebar-note { color:var(--muted); font-size:.78rem; line-height:1.5; }
@media (max-width:800px) { .block-container { padding:1.1rem 1rem 4.5rem; } .hero { padding-top:.2rem; } }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------

@st.cache_resource(show_spinner=False)
def cached_video_index(video_id: str, _progress_callback: Any = None) -> VideoIndex:
    return build_video_index(video_id, progress_callback=_progress_callback)


@st.cache_resource(show_spinner=False)
def cached_llm(api_key: str, model_name: str):
    return create_llm(api_key=api_key, model_name=model_name)


def reset_conversation() -> None:
    st.session_state.chat_history = []


def show_sources(sources: list[dict[str, Any]], video_id: str) -> None:
    if not sources:
        return
    with st.expander(f"Sources · {len(sources)} transcript excerpts"):
        for source in sources:
            start = int(source.get("start_seconds", 0))
            timestamp = source.get("timestamp", "00:00")
            st.markdown(f"[{timestamp} · Open at this point]({youtube_watch_url(video_id, start)})")
            st.caption(source["text"])


def render_message(message: dict[str, Any], video_id: str | None) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and video_id:
            show_sources(message.get("sources", []), video_id)


# ---------------- SESSION STATE ----------------

st.session_state.setdefault("video_id", None)
st.session_state.setdefault("video_index", None)
st.session_state.setdefault("index_stats", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("pending_question", None)
st.session_state.setdefault("processing_error", None)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">▶</div>'
        '<div class="brand-name">TubeMind</div></div>',
        unsafe_allow_html=True,
    )
    st.caption("VIDEO INTELLIGENCE WORKSPACE")
    st.divider()

    video_input = st.text_input(
        "YouTube video",
        placeholder="Paste a URL or 11-character ID",
        help="Public videos with English or Telugu captions are supported.",
    )

    process_clicked = st.button(
        "Process video",
        type="primary",
        icon="▶",
        use_container_width=True,
    )

    st.divider()

    if st.session_state.video_id:
        st.markdown("**INDEX STATUS**")
        st.success("Transcript indexed", icon="✅")
        stats = st.session_state.index_stats or {}
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Chunks", stats.get("chunk_count", 0))
        metric_col2.metric("Captions", stats.get("segment_count", 0))
        st.markdown(f"[Open on YouTube ↗]({youtube_watch_url(st.session_state.video_id)})")
        if st.button("New chat", icon="🔄", use_container_width=True):
            reset_conversation()
            st.rerun()
    else:
        st.markdown(
            '<div class="sidebar-note">Paste a video link to index its '
            'captions and start a grounded conversation.</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<div class="sidebar-note">Built with Streamlit, LangChain, '
        'FAISS, Hugging Face, and Gemini.</div>',
        unsafe_allow_html=True,
    )


# ---------------- VIDEO PROCESSING ----------------

if process_clicked:
    new_video_id = extract_video_id(video_input)

    if not new_video_id:
        st.session_state.processing_error = "Enter a valid YouTube URL or video ID."
    else:
        st.session_state.processing_error = None
        progress_steps = {
            "Fetching captions": 15,
            "Preparing timestamped chunks": 40,
            "Loading embedding model": 65,
            "Building FAISS search index": 85,
        }
        try:
            with st.status("Preparing your video index...", expanded=True) as status:
                progress = st.progress(0)

                def update_progress(label: str) -> None:
                    status.update(label=label, state="running")
                    progress.progress(progress_steps.get(label, 10))

                index = cached_video_index(new_video_id, update_progress)
                progress.progress(100)
                status.update(label="Video is ready to chat", state="complete", expanded=False)
            st.session_state.video_id = new_video_id
            st.session_state.video_index = index.vector_store
            st.session_state.index_stats = {
                "chunk_count": index.chunk_count,
                "segment_count": index.segment_count,
            }
            st.session_state.chat_history = []
        except (TranscriptsDisabled, NoTranscriptFound):
            st.session_state.processing_error = "No English or Telugu captions were found for this video."
        except VideoUnavailable:
            st.session_state.processing_error = "This video is unavailable or private."
        except YouTubeTranscriptApiException as exc:
            st.session_state.processing_error = f"YouTube could not provide captions: {exc}"
        except Exception as exc:
            st.session_state.processing_error = f"Could not index this video ({type(exc).__name__}): {exc}"

if st.session_state.processing_error:
    st.error(st.session_state.processing_error, icon="⚠️")


# ---------------- MAIN PAGE ----------------

st.markdown("""
<div class="hero"><div class="eyebrow">TRANSCRIPT-POWERED AI</div>
<h1>Make every video searchable.</h1>
<p>Ask focused questions and explore answers grounded in the original captions.</p></div>
""", unsafe_allow_html=True)

chat_col, video_col = st.columns([1.55, 1], gap="large")
with chat_col:
    if st.session_state.video_id:
        st.markdown('<div class="workspace-title">Conversation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="workspace-subtitle">Source: '
            f'<a href="{youtube_watch_url(st.session_state.video_id)}" target="_blank">'
            f'youtube.com/watch?v={st.session_state.video_id}</a></div>',
            unsafe_allow_html=True,
        )
        if not st.session_state.chat_history:
            st.markdown(
                '<div class="empty-panel"><div class="empty-index">✳</div>'
                '<strong>Your video is indexed.</strong><br>'
                '<span style="color:#8e9a95">Ask a question below, or start with one of these.</span></div>',
                unsafe_allow_html=True,
            )
            for suggested_question in (
                "Summarize the main ideas",
                "What are the most important details?",
                "What does the speaker conclude?",
            ):
                if st.button(suggested_question, key=f"suggest-{suggested_question}", use_container_width=True):
                    st.session_state.pending_question = suggested_question
                    st.rerun()
        for chat_message in st.session_state.chat_history:
            render_message(chat_message, st.session_state.video_id)
    else:
        st.markdown(
            '<div class="empty-panel"><div class="empty-index">▶</div>'
            '<div class="eyebrow">YOUR VIDEO, UNDERSTOOD</div>'
            '<h2 style="font-family:Space Grotesk,sans-serif;margin:.45rem 0">Start with a video.</h2>'
            '<p style="color:#8e9a95">TubeMind indexes its captions, finds relevant passages, '
            'and answers with links back to the source.</p></div>',
            unsafe_allow_html=True,
        )
        st.markdown("**Try asking**")
        for suggestion in ("Give me a concise summary", "What are the key takeaways?", "Find a specific topic in the video"):
            st.markdown(f"`{suggestion}`")

with video_col:
    st.markdown('<div class="workspace-title">Video source</div>', unsafe_allow_html=True)
    st.markdown('<div class="workspace-subtitle">Preview and transcript-backed references</div>', unsafe_allow_html=True)
    if st.session_state.video_id:
        st.video(youtube_watch_url(st.session_state.video_id))
        stats = st.session_state.index_stats or {}
        st.caption(
            f"Indexed {stats.get('segment_count', 0):,} caption segments into "
            f"{stats.get('chunk_count', 0):,} searchable chunks."
        )
    else:
        st.markdown(
            '<div class="empty-panel" style="aspect-ratio:16/9;display:grid;place-items:center;'
            'text-align:center;color:#8e9a95"><div><div style="font-size:2rem;color:#f18468">▶</div>'
            '<div>Video preview appears here</div></div></div>',
            unsafe_allow_html=True,
        )

# ---------------- CHAT HISTORY ----------------

if st.session_state.video_index is not None:
    question = st.chat_input("Ask something about this video...")
    question = question or st.session_state.pending_question
    st.session_state.pending_question = None
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        render_message(st.session_state.chat_history[-1], st.session_state.video_id)
        try:
            settings = get_settings()
            try:
                secret_key = st.secrets.get("GOOGLE_API_KEY")
            except (FileNotFoundError, KeyError, StreamlitSecretNotFoundError):
                secret_key = None
            api_key = get_google_api_key(secret_key)
            llm = cached_llm(api_key, settings.chat_model)
            with st.chat_message("assistant"):
                with st.spinner("Searching captions and drafting an answer..."):
                    answer, source_docs = answer_question(
                        question=question,
                        retriever=st.session_state.video_index.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": settings.retrieval_count},
                        ),
                        llm=llm,
                        history=st.session_state.chat_history[:-1],
                    )
                source_items = [
                    {
                        "text": document.page_content,
                        "start_seconds": document.metadata.get("start_seconds", 0),
                        "timestamp": document.metadata.get("timestamp", "00:00"),
                    }
                    for document in source_docs
                ]
                st.markdown(answer)
                show_sources(source_items, st.session_state.video_id)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "sources": source_items}
            )
        except ValueError as exc:
            st.error(str(exc), icon="⚠️")
            st.session_state.chat_history.pop()
        except Exception as exc:
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                message = "Gemini rate limit or quota reached. Check your Google AI API quota and try again."
            st.error(f"Could not generate an answer ({type(exc).__name__}): {message}", icon="⚠️")
            st.session_state.chat_history.pop()
else:
    st.chat_input("Process a video to start a conversation", disabled=True)
