"""Whisper transcription provider using faster-whisper on CPU."""
import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from ophanim.core.audio import extract_audio, has_audio_stream, cleanup_wav

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcribed segment with timing."""
    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class Transcript:
    """Full transcription result."""
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = ""
    duration_seconds: float = 0.0

    @property
    def text(self) -> str:
        """Full concatenated text."""
        return " ".join(s.text for s in self.segments)

    @property
    def segment_count(self) -> int:
        return len(self.segments)


class WhisperProvider:
    """
    Speech-to-text provider using faster-whisper.

    Runs on CPU with int8 quantization to avoid GPU VRAM contention
    with LM Studio's vision model.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.model_size = self.config.get("model_size", "base")
        self.device = self.config.get("device", "cpu")
        self.compute_type = self.config.get("compute_type", "int8")
        self._model = None

    def _ensure_model(self):
        """Lazy-load the Whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel
            logger.info(
                f"Loading Whisper model '{self.model_size}' on {self.device} "
                f"(compute={self.compute_type})"
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(self, video_path: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe audio from a video file.

        Steps:
        1. Extract audio to WAV using ffmpeg
        2. Transcribe with faster-whisper
        3. Clean up temp WAV

        Args:
            video_path: Path to video file
            language: Optional language code (e.g., "en"). Auto-detect if None.

        Returns:
            Transcript with segments, language, duration
        """
        if not has_audio_stream(video_path):
            logger.warning(f"No audio stream in {video_path}")
            return Transcript(duration_seconds=0.0)

        # Extract audio to temp WAV
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            wav_path = extract_audio(video_path, output_path=tmp_path)
            return self._transcribe_wav(wav_path, language)
        finally:
            cleanup_wav(tmp_path)

    def transcribe_audio(self, audio_path: str, language: Optional[str] = None) -> Transcript:
        """
        Transcribe from an already-extracted WAV file.

        Args:
            audio_path: Path to WAV file (16kHz mono PCM)
            language: Optional language code
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self._transcribe_wav(audio_path, language)

    def _transcribe_wav(self, wav_path: str, language: Optional[str] = None) -> Transcript:
        """Internal: transcribe a WAV file."""
        self._ensure_model()

        segments, info = self._model.transcribe(
            wav_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
            ),
        )

        transcript = Transcript(
            language=info.language if info else "",
            duration_seconds=info.duration if info else 0.0,
        )

        for seg in segments:
            transcript.segments.append(TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                confidence=seg.avg_logprob if hasattr(seg, 'avg_logprob') else 0.0,
            ))

        logger.info(
            f"Transcribed {len(transcript.segments)} segments "
            f"({transcript.duration_seconds:.1f}s, lang={transcript.language})"
        )

        return transcript

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self):
        """Unload the Whisper model to free memory."""
        self._model = None
        import gc
        gc.collect()
