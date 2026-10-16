import unittest
from unittest.mock import patch

from src.indexer import create_transcript_documents
from src.transcript import (
    TranscriptSegment,
    extract_video_id,
    format_timestamp,
    fetch_transcript,
    youtube_watch_url,
)


class ExtractVideoIdTests(unittest.TestCase):
    def test_accepts_raw_video_id(self):
        self.assertEqual(extract_video_id("WxTQybN5_WI"), "WxTQybN5_WI")

    def test_accepts_watch_and_short_urls(self):
        self.assertEqual(
            extract_video_id("https://www.youtube.com/watch?v=WxTQybN5_WI&t=3"),
            "WxTQybN5_WI",
        )
        self.assertEqual(
            extract_video_id("https://youtu.be/WxTQybN5_WI"), "WxTQybN5_WI"
        )

    def test_rejects_untrusted_hosts_and_malformed_ids(self):
        self.assertIsNone(extract_video_id("https://example.com/watch?v=WxTQybN5_WI"))
        self.assertIsNone(extract_video_id("too-short"))


class TranscriptHelpersTests(unittest.TestCase):
    @patch("src.transcript.YouTubeTranscriptApi")
    def test_fetch_uses_supported_languages_and_preserves_timing(self, api_class):
        fetched = api_class.return_value.fetch.return_value
        fetched.to_raw_data.return_value = [
            {"text": "  Caption text  ", "start": 2.5, "duration": 3.0}
        ]

        segments = fetch_transcript("WxTQybN5_WI")

        api_class.return_value.fetch.assert_called_once_with(
            "WxTQybN5_WI", languages=["en", "te"]
        )
        self.assertEqual(segments, [TranscriptSegment("Caption text", 2.5, 3.0)])

    def test_formats_timestamps_and_seek_urls(self):
        self.assertEqual(format_timestamp(83), "01:23")
        self.assertEqual(format_timestamp(3661), "1:01:01")
        self.assertEqual(
            youtube_watch_url("WxTQybN5_WI", 83),
            "https://www.youtube.com/watch?v=WxTQybN5_WI&t=83s",
        )

    def test_chunks_keep_caption_timestamps(self):
        documents = create_transcript_documents(
            "WxTQybN5_WI",
            [
                TranscriptSegment("The first caption explains the topic.", 12.4, 3.0),
                TranscriptSegment("The second caption gives another detail.", 16.0, 4.0),
            ],
        )
        self.assertTrue(documents)
        self.assertEqual(documents[0].metadata["start_seconds"], 12)
        self.assertEqual(documents[0].metadata["timestamp"], "00:12")
        self.assertIn("t=12s", documents[0].metadata["source_url"])

    def test_rejects_empty_transcripts(self):
        with self.assertRaisesRegex(ValueError, "transcript is empty"):
            create_transcript_documents("WxTQybN5_WI", [])


if __name__ == "__main__":
    unittest.main()