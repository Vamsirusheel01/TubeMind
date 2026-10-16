from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi


VIDEO_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{11}\Z")


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float
    duration: float


def extract_video_id(value: str) -> str | None:
    """Return a valid 11-character YouTube video ID from a URL or raw ID."""
    candidate = value.strip()
    if VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    if host == "youtu.be":
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        else:
            match = re.match(r"^/(?:embed|shorts|live)/([^/?]+)", parsed.path)
            video_id = match.group(1) if match else ""
    else:
        return None

    return video_id if VIDEO_ID_PATTERN.fullmatch(video_id) else None


def youtube_watch_url(video_id: str, start_seconds: int | None = None) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    if start_seconds is not None and start_seconds > 0:
        return f"{url}&t={start_seconds}s"
    return url


def format_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02}:{remaining_seconds:02}"
    return f"{minutes:02}:{remaining_seconds:02}"


def fetch_transcript(
    video_id: str,
    languages: tuple[str, ...] = ("en", "te"),
) -> list[TranscriptSegment]:
    """Fetch caption entries while retaining their original timestamps."""
    fetched = YouTubeTranscriptApi().fetch(video_id, languages=list(languages))
    return [
        TranscriptSegment(
            text=str(item["text"]).strip(),
            start=float(item["start"]),
            duration=float(item["duration"]),
        )
        for item in fetched.to_raw_data()
        if str(item.get("text", "")).strip()
    ]