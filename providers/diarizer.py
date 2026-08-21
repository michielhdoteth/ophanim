"""Speaker diarization provider - identifies who speaks when."""
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A diarized segment identifying a speaker."""
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    """Full diarization result."""
    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0
    speakers: list[str] = field(default_factory=list)


class DiarizerProvider:
    """
    Speaker diarization using the `diarize` library (FoxNoseTech).

    CPU-only, no HF token required. ~4.8% DER, 8x realtime on CPU.
    Install: pip install diarize
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._lib = None

    def _ensure_lib(self):
        """Lazy-load the diarize library."""
        if self._lib is None:
            try:
                from diarize import diarize as diarize_fn
                self._lib = diarize_fn
            except ImportError:
                raise ImportError(
                    "Speaker diarization requires the 'diarize' package.\n"
                    "Install it with: pip install diarize"
                )

    def diarize(self, audio_path: str,
                min_speakers: Optional[int] = None,
                max_speakers: Optional[int] = None) -> DiarizationResult:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)

        Returns:
            DiarizationResult with speaker segments
        """
        self._ensure_lib()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        kwargs = {}
        if min_speakers is not None:
            kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            kwargs["max_speakers"] = max_speakers

        logger.info(f"Running diarization on {audio_path}")
        result = self._lib(audio_path, **kwargs)

        # Convert to our dataclass
        segments = [
            SpeakerSegment(start=s.start, end=s.end, speaker=s.speaker)
            for s in result.segments
        ]

        speakers = list(set(s.speaker for s in segments))
        speakers.sort()

        diar_result = DiarizationResult(
            segments=segments,
            num_speakers=result.num_speakers,
            speakers=speakers,
        )

        logger.info(f"Diarization complete: {len(segments)} segments, {len(speakers)} speakers")
        return diar_result


def merge_transcript_with_diarization(
    transcript_segments: list,
    diarization_segments: list[SpeakerSegment],
    fill_nearest: bool = True,
) -> list:
    """
    Merge whisper transcript segments with speaker diarization output.

    For each transcript segment, finds the diarization segment with the
    greatest temporal overlap and assigns that speaker label.

    Args:
        transcript_segments: List of TranscriptSegment from faster-whisper
        diarization_segments: List of SpeakerSegment from diarizer
        fill_nearest: If True, assign nearest speaker when no overlap exists

    Returns:
        List of TranscriptSegment with speaker labels filled in
    """
    diarization_segments = sorted(diarization_segments, key=lambda x: x.start)
    merged = []

    for seg in transcript_segments:
        seg_start = seg.start
        seg_end = seg.end
        speaker_overlap: dict[str, float] = {}

        for dia in diarization_segments:
            # Calculate temporal intersection
            intersection = min(dia.end, seg_end) - max(dia.start, seg_start)
            if intersection <= 0:
                continue

            speaker = dia.speaker
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + intersection

        if speaker_overlap:
            # Assign speaker with greatest overlap
            speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
        elif fill_nearest and diarization_segments:
            # Fallback: find nearest diarization segment by midpoint
            midpoint = (seg_start + seg_end) / 2
            nearest = min(
                diarization_segments,
                key=lambda x: abs(((x.start + x.end) / 2) - midpoint),
            )
            speaker = nearest.speaker
        else:
            speaker = ""

        # Create new segment with speaker label
        merged.append(type(seg)(
            start=seg.start,
            end=seg.end,
            text=seg.text,
            confidence=seg.confidence,
            speaker=speaker,
        ))

    return merged
