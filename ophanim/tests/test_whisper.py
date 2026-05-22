"""Tests for Whisper provider."""
import pytest
from pathlib import Path
from ophanim.providers.whisper import WhisperProvider, Transcript, TranscriptSegment


class TestTranscript:
    def test_empty_transcript(self):
        t = Transcript()
        assert t.text == ""
        assert t.segment_count == 0

    def test_with_segments(self):
        t = Transcript(
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="hello", confidence=0.9),
                TranscriptSegment(start=1.0, end=2.0, text="world", confidence=0.8),
            ],
            language="en",
            duration_seconds=2.0,
        )
        assert t.text == "hello world"
        assert t.segment_count == 2


class TestWhisperProvider:
    def test_init(self):
        provider = WhisperProvider({"model_size": "tiny"})
        assert provider.model_size == "tiny"
        assert not provider.is_loaded

    def test_transcribe_no_audio(self, tmp_path):
        """transcribe() should return empty Transcript when video has no audio."""
        import subprocess

        # Create a minimal video with no audio stream
        video_path = tmp_path / "no_audio.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1",
             "-c:v", "libx264", "-an", str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        if not video_path.exists():
            pytest.skip("Could not create test video")

        provider = WhisperProvider({"model_size": "tiny"})
        result = provider.transcribe(str(video_path))
        assert isinstance(result, Transcript)
        assert result.segment_count == 0
        assert result.text == ""

    def test_transcribe_real_video(self):
        """Integration test with real video from Downloads/Videos."""
        dl = Path(r"C:\Users\michi\Downloads\Videos")
        if not dl.exists():
            pytest.skip("No test videos directory")
        videos = list(dl.glob("*.mp4"))
        if not videos:
            pytest.skip("No test video available")

        provider = WhisperProvider({"model_size": "tiny"})
        try:
            result = provider.transcribe(str(videos[0]))
            assert isinstance(result, Transcript)
            assert result.duration_seconds > 0
            assert len(result.language) > 0
        finally:
            provider.unload()
