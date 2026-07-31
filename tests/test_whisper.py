"""Tests for Whisper provider."""
import pytest
from pathlib import Path
from openvision.providers.whisper import WhisperProvider, Transcript, TranscriptSegment


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

    def test_transcribe_real_video(self, tmp_path):
        """Integration test: transcribe a short generated video (mocked model)."""
        import subprocess

        video_path = tmp_path / "real_video.mp4"
        # Create 2s video with silent audio track
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=2",
             "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
             "-shortest", "-c:v", "libx264", "-c:a", "aac",
             str(video_path)],
            capture_output=True, text=True, timeout=30,
        )
        if not video_path.exists():
            pytest.skip("Could not create test video")

        from unittest.mock import MagicMock

        # Mock the faster_whisper model to avoid large download / slow inference
        mock_model = MagicMock()
        # Simulate faster_whisper transcribe API: returns (segments_iterator, info)
        mock_seg = MagicMock()
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.text = " hello "
        mock_seg.avg_logprob = -0.5
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.duration = 2.0
        mock_model.transcribe.return_value = ([mock_seg], mock_info)

        provider = WhisperProvider({"model_size": "tiny"})
        provider._model = mock_model  # Inject mock so _ensure_model is skipped
        result = provider.transcribe(str(video_path))
        assert isinstance(result, Transcript)
        assert result.text == "hello"
        assert result.language == "en"
        assert result.duration_seconds == 2.0
        provider.unload()
