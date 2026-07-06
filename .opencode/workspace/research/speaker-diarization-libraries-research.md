# Research: Speaker Diarization Libraries for Ophanim
- Component: core (transcription pipeline)
- Created: 2026-07-05
- Status: complete

## Summary

For ophanim's local-first architecture (faster-whisper + GPU-aware auto-detect), the best fit is a two-tier strategy: **`diarize` (FoxNoseTech)** as the primary CPU diarizer for its zero-config, Apache 2.0 license, and no HF token requirement, with **`pyannote.audio` 3.1** as an optional GPU-accelerated fallback for users who want higher accuracy and already have a HF token. WhisperX should NOT replace faster-whisper -- it wraps it but adds unnecessary complexity for ophanim's use case.

---

## Findings

### 1. pyannote.audio 3.x / 4.x

**Current state (2026):**
- pyannote-audio library is at v4.0.7 (MIT license on PyPI)
- Two main pretrained pipelines available:
  - `pyannote/speaker-diarization-3.1` -- legacy, MIT license, needs HF token + model access acceptance
  - `pyannote/speaker-diarization-community-1` -- new best open-source, CC-BY-4.0 license, needs HF token + model access acceptance
- pyannote 4.0 introduced VBx clustering (replacing agglomerative hierarchical clustering), significantly improving speaker counting and assignment
- The `community-1` model returns both regular and "exclusive" diarization (simplified merge with ASR)

**API:**
```python
from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token="HF_TOKEN"
)
# CPU by default; optional GPU:
# pipeline.to(torch.device("cuda"))

diarization = pipeline("audio.wav")

# pyannote 4.x returns DiarizeOutput
# Access via .speaker_diarization for the Annotation object
ann = getattr(diarization, "speaker_diarization", diarization)

for turn, _, speaker in ann.itertracks(yield_label=True):
    print(f"[{turn.start:.1f}s -> {turn.end:.1f}s] {speaker}")
```

**Key requirements:**
- HuggingFace account + access token (mandatory)
- Must accept model user conditions on HF for each model
- Models can be cloned for fully offline use after initial download
- Supports `num_speakers`, `min_speakers`, `max_speakers` parameters
- Progress hooks available: `pipeline("audio.wav", hook=hook)`

**CPU vs GPU:**
- Runs on CPU by default (slower, ~0.86 RTF)
- GPU acceleration via `pipeline.to(torch.device("cuda"))` -- significantly faster
- No CUDA-optional auto-detect like ophanim has; user must explicitly choose

**Accuracy (DER on VoxConverse dev, lower is better):**
- community-1: ~11.2% DER
- 3.1 (legacy): ~11.2% DER
- precision-2 (commercial): ~8.5% DER

**License:** Library is MIT. Community-1 model is CC-BY-4.0. Legacy 3.1 model is MIT.

---

### 2. WhisperX

**Current state:** v3.8.5 (April 2025), BSD-2-Clause license

**What it is:** A Python layer on top of faster-whisper that adds:
1. Voice activity detection preprocessing (Silero-VAD)
2. Word-level forced alignment (wav2vec2 models, sub-100ms timestamps)
3. Speaker diarization (pyannote.audio integration)

**Key findings:**
- WhisperX WRAPS faster-whisper -- it does not replace it. It uses faster-whisper internally for transcription.
- Adds ~20-60% overhead depending on features used (alignment + diarization)
- Requires HuggingFace token for diarization (same pyannote dependency)
- The diarization is the same pyannote pipeline, just bundled

**Should ophanim use WhisperX?**
- **NO.** WhisperX would replace ophanim's carefully built WhisperProvider with a black box.
- Ophanim already has faster-whisper with custom VAD parameters, GPU auto-detect, and a clean TranscriptSegment dataclass.
- WhisperX's value is diarization + alignment, but ophanim can add diarization independently.
- WhisperX's alignment is useful for word-level timestamps, but ophanim's TranscriptSegment uses segment-level timestamps which are sufficient.

**If you wanted WhisperX's alignment feature:** It's a separate concern from diarization and could be added later as an optional post-processing step.

---

### 3. simple-diarizer

**Current state:** v0.0.13, last meaningful update unclear (155 GitHub stars)

**What it is:** Simplified diarization using SpeechBrain pretrained models (X-Vector or ECAPA-TDNN embeddings) + spectral clustering.

**API:**
```python
from simple_diarizer.diarizer import Diarizer

diar = Diarizer(embed_model='xvec', cluster_method='sc')
segments = diar.diarize("audio.wav", num_speakers=2)
# Returns list of (start, end, speaker) tuples
```

**Pros:**
- Simple API, easy to integrate
- No HF token needed
- Uses SpeechBrain models (well-maintained)

**Cons:**
- Requires `num_speakers` to be known upfront (or use threshold-based detection which is less reliable)
- No automatic speaker count estimation built-in (unlike pyannote and diarize)
- Accuracy significantly lower than pyannote 3.x on benchmarks
- Project appears less actively maintained
- Dependencies: speechbrain, torchaudio, scikit-learn, matplotlib, pandas

**Verdict:** Not recommended. The lack of automatic speaker count estimation is a dealbreaker for a general-purpose tool. simple-diarizer is a stepping stone, not a destination.

---

### 4. NVIDIA NeMo

**Current state:** Active development, part of NVIDIA's NeMo toolkit

**What it is:** Enterprise-grade ASR + speaker diarization framework. Supports both cascaded (modular) and end-to-end (Sortformer) diarization.

**Key findings:**
- Massive dependency footprint (NeMo toolkit, NVIDIA-specific CUDA stacks)
- Designed for NVIDIA GPU environments; CPU support exists but is impractical
- Complex configuration (YAML-based, PyTorch Lightning)
- Models hosted on NGC (NVIDIA's model registry), not PyPI-friendly
- Excellent accuracy but massive overkill for ophanim's use case
- Not pip-installable in a lightweight way

**CPU vs GPU:**
- GPU-primary design; CPU inference possible but very slow
- Requires CUDA toolkit + cuDNN for GPU

**Verdict:** **Overkill.** NeMo is designed for enterprise speech pipelines at scale. The dependency weight alone would triple ophanim's install size. The accuracy advantage doesn't justify the complexity for podcast/interview diarization.

---

### 5. diarize (FoxNoseTech)

**Current state:** v0.1.2 (2026-02-28), Apache 2.0 license, actively maintained

**What it is:** A focused, CPU-only diarization library. 4-stage pipeline:
1. Silero VAD (MIT) -- speech detection
2. WeSpeaker ResNet34-LM (Apache 2.0) -- 256-dim speaker embeddings via ONNX
3. GMM BIC + silhouette refinement -- automatic speaker count estimation
4. Spectral Clustering (scikit-learn, BSD) + temporal smoothing -- speaker labels

**API:**
```python
from diarize import diarize

result = diarize("meeting.wav")
# Or with constraints:
result = diarize("meeting.wav", min_speakers=2, max_speakers=5)

print(result.num_speakers)      # 3
print(result.speakers)          # ['SPEAKER_00', 'SPEAKER_01', 'SPEAKER_02']

for seg in result.segments:
    print(f"[{seg.start:.1f}s -> {seg.end:.1f}s] {seg.speaker}")

# Export
result.to_rttm("output.rttm")
result.to_list()  # [{"start": float, "end": float, "speaker": str}, ...]
```

**Key advantages for ophanim:**
- **Zero config:** `pip install diarize` -- no HF tokens, no model acceptance, no account signup
- **CPU-only by design:** Aligns perfectly with ophanim's "local-first" philosophy
- **Auto speaker count:** GMM BIC estimation, no need to know speaker count upfront
- **Better accuracy than pyannote free:** ~4.8% DER vs ~11.2% DER on VoxConverse dev
- **8x faster than real-time on CPU:** RTF 0.12 vs pyannote's 0.86
- **Clean output format:** `DiarizeResult` with `.segments`, `.to_list()`, `.to_rttm()`
- **All permissive licenses:** Apache 2.0, MIT, BSD throughout
- **Lightweight deps:** torch (pinned range), scikit-learn, silero-vad, wespeakerruntime, pydantic, soundfile

**CPU vs GPU:**
- CPU-only by design (no GPU support, no need for it at this speed)
- 8x faster than real-time on CPU is sufficient for podcasts/interviews

**Accuracy:**
- ~4.8% weighted DER on VoxConverse dev (best among free/open-source options)
- ~4.8% vs pyannote community-1's ~11.2% (2.3x better)

**License:** Apache 2.0. All dependencies are MIT/BSD/Apache 2.0.

---

## Comparison Table

| Feature | diarize (FoxNoseTech) | pyannote 3.1 / community-1 | simple-diarizer | WhisperX | NeMo |
|---|---|---|---|---|---|
| **License** | Apache 2.0 | MIT (lib) / CC-BY-4.0 (community-1 model) | MIT (SpeechBrain deps) | BSD-2-Clause | Apache 2.0 (NeMo) |
| **HF Token Required** | No | Yes | No | Yes (for diarization) | No (NGC) |
| **CPU Support** | Yes (CPU-only) | Yes (default) | Yes | Yes | Slow |
| **GPU Support** | No (by design) | Yes (optional) | Yes (via torch) | Yes | Yes (primary) |
| **Auto Speaker Count** | Yes (GMM BIC) | Yes | No (needs num_speakers) | Yes (via pyannote) | Yes |
| **DER (VoxConverse)** | ~4.8% | ~11.2% (community-1) | Higher (not benchmarked) | Same as pyannote | Competitive |
| **CPU Speed (RTF)** | 0.12 (8x realtime) | 0.86 (1.2x realtime) | Moderate | 0.5-0.8 (estimated) | Very slow |
| **Install Complexity** | `pip install diarize` | `pip install pyannote.audio` + HF setup | `pip install simple-diarizer` | `pip install whisperx` | Massive (NeMo toolkit) |
| **Offline After Download** | Yes | Yes (clone repo) | Yes | Yes (clone) | Yes |
| **Output Format** | DiarizeResult (segments, to_list, to_rttm) | Annotation (itertracks) | List of tuples | Dict with segments | Custom |
| **Overlapping Speech** | No (single speaker per segment) | Yes (powerset model) | No | Yes (via pyannote) | Yes |
| **Python Version** | >=3.9 | >=3.9 | >=3.7 | >=3.8 | >=3.9 |

---

## How to Merge Diarization with Whisper Transcript Segments

### The Standard Approach (Overlap-Based)

This is the canonical approach used by pyannote's own documentation, WhisperX, and virtually every whisper+diarization integration:

```python
from ophanim.providers.whisper import TranscriptSegment

def merge_transcript_with_diarization(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[dict],  # [{"start": float, "end": float, "speaker": str}]
    fill_nearest: bool = True,
) -> list[dict]:
    """
    Merge whisper transcript segments with speaker diarization output.

    For each transcript segment, finds the diarization segment with the
    greatest temporal overlap and assigns that speaker label.

    Args:
        transcript_segments: List of TranscriptSegment from faster-whisper
        diarization_segments: List of {"start", "end", "speaker"} dicts
        fill_nearest: If True, assign nearest speaker when no overlap exists

    Returns:
        List of {"start", "end", "text", "speaker"} dicts
    """
    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
    merged = []

    for seg in transcript_segments:
        seg_start = seg.start
        seg_end = seg.end
        speaker_overlap: dict[str, float] = {}

        for dia in diarization_segments:
            # Calculate temporal intersection
            intersection = min(dia["end"], seg_end) - max(dia["start"], seg_start)
            if intersection <= 0:
                continue

            speaker = dia["speaker"]
            speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + intersection

        if speaker_overlap:
            # Assign speaker with greatest overlap
            speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
        elif fill_nearest and diarization_segments:
            # Fallback: find nearest diarization segment by midpoint
            midpoint = (seg_start + seg_end) / 2
            nearest = min(
                diarization_segments,
                key=lambda x: abs(((x["start"] + x["end"]) / 2) - midpoint),
            )
            speaker = nearest["speaker"]
        else:
            speaker = "UNKNOWN"

        merged.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg.text,
            "speaker": speaker,
        })

    return merged
```

### Integration with diarize library

```python
from diarize import diarize as diarize_audio
from ophanim.providers.whisper import WhisperProvider, TranscriptSegment

def transcribe_with_speakers(
    audio_path: str,
    whisper_config: dict | None = None,
    diarize_config: dict | None = None,
) -> list[dict]:
    """Transcribe audio with speaker labels."""
    # Step 1: Transcribe with faster-whisper
    whisper = WhisperProvider(whisper_config or {})
    transcript = whisper.transcribe_audio(audio_path)

    # Step 2: Diarize with diarize library
    result = diarize_audio(audio_path, **(diarize_config or {}))
    dia_segments = [{"start": s.start, "end": s.end, "speaker": s.speaker}
                    for s in result.segments]

    # Step 3: Merge
    return merge_transcript_with_diarization(
        transcript.segments, dia_segments, fill_nearest=True
    )
```

### Integration with pyannote (if chosen)

```python
from pyannote.audio import Pipeline
import torch
from ophanim.providers.whisper import WhisperProvider

def transcribe_with_speakers_pyannote(
    audio_path: str,
    hf_token: str,
    device: str = "auto",
) -> list[dict]:
    """Transcribe with pyannote diarization."""
    # Step 1: Transcribe
    whisper = WhisperProvider({"device": device})
    transcript = whisper.transcribe_audio(audio_path)

    # Step 2: Diarize
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=hf_token,
    )
    if device == "cuda":
        pipeline.to(torch.device("cuda"))

    diarization = pipeline(audio_path)

    # Handle pyannote 4.x DiarizeOutput
    ann = getattr(diarization, "speaker_diarization", diarization)

    dia_segments = []
    for turn, _, speaker in ann.itertracks(yield_label=True):
        dia_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    # Step 3: Merge
    return merge_transcript_with_diarization(
        transcript.segments, dia_segments, fill_nearest=True
    )
```

### Edge Cases to Handle

1. **Segment spanning speaker boundary:** A whisper segment might span two speakers. The overlap-based approach assigns the dominant speaker. For higher precision, split the segment at the diarization boundary.

2. **No overlap at all:** When `fill_nearest=True`, the nearest diarization segment by midpoint is used. When `fill_nearest=False`, speaker is "UNKNOWN".

3. **pyannote 4.x API change:** pyannote 4.x returns `DiarizeOutput` instead of raw `Annotation`. Use `getattr(diarization, "speaker_diarization", diarization)` for compatibility.

4. **Consecutive same-speaker segments:** After merging, optionally merge adjacent segments with the same speaker to produce cleaner output.

---

## Recommendation

### Primary: `diarize` (FoxNoseTech)

**Why it's the best fit for ophanim:**

1. **Philosophy alignment:** Local-first, CPU-only, no accounts/tokens. Matches ophanim's "no cloud APIs" requirement perfectly.
2. **Zero friction:** `pip install diarize` -- no HF tokens, no model acceptance gates, no account creation. Users can start immediately.
3. **Better accuracy:** ~4.8% DER vs pyannote's ~11.2% on VoxConverse. For podcasts and interviews, this is excellent.
4. **Speed:** 8x faster than real-time on CPU means a 60-minute podcast diarizes in ~7.5 minutes on CPU. No GPU needed.
5. **Clean API:** `DiarizeResult` with `.segments`, `.to_list()`, `.to_rttm()` maps directly to ophanim's dataclass pattern.
6. **License:** Apache 2.0 with all-permissive deps. No license concerns.
7. **Lightweight:** Doesn't pull in the full pyannote/HuggingFace ecosystem.

### Secondary (Optional): `pyannote.audio` 3.1 / community-1

Keep as an optional provider for users who:
- Want GPU-accelerated diarization
- Already have a HuggingFace token
- Need overlapping speech detection
- Want the "exclusive" diarization feature from community-1

### Implementation Plan

1. Create `src/ophanim/providers/diarizer.py` with a `DiarizerProvider` base class
2. Implement `DiarizeProvider` using the `diarize` library as default
3. Implement `PyannoteDiarizerProvider` as optional (requires HF token)
4. Add `SpeakerSegment` dataclass: `start, end, speaker`
5. Add `merge_transcript_with_diarization()` utility function
6. Update `TranscriptSegment` to optionally include `speaker: str` field
7. Add `diarize` to dependencies in `pyproject.toml`
8. Make `pyannote.audio` an optional extra: `pip install ophanim[pyannote]`

### Risk Assessment

| Risk | Mitigation |
|---|---|
| `diarize` is new (Feb 2026), limited track record | Benchmark on ophanim's target content; fallback to pyannote |
| No GPU support in `diarize` | CPU is sufficient for podcast/interview length content |
| No overlapping speech detection in `diarize` | Most podcast/interview content has minimal overlap; add warning |
| pyannote 4.x API breaking change | Use `getattr` compatibility shim in PyannoteDiarizerProvider |
| HF token requirement for pyannote | Make it optional; clear error message if missing |

---

## Sources

1. pyannote/pyannote-audio GitHub: https://github.com/pyannote/pyannote-audio
2. pyannote/speaker-diarization-community-1 HF: https://huggingface.co/pyannote/speaker-diarization-community-1
3. pyannote/speaker-diarization-3.1 HF: https://huggingface.co/pyannote/speaker-diarization-3.1
4. pyannote-audio v3.4.0 PyPI: https://pypi.org/project/pyannote-audio/3.4.0/
5. pyannote 4.0.0 Changelog: https://github.com/pyannote/pyannote-audio/blob/main/CHANGELOG.md
6. Community-1 blog post: https://www.pyannote.ai/blog/community-1
7. WhisperX GitHub: https://github.com/m-bain/whisperX
8. WhisperX vs Faster-Whisper 2026 comparison: https://aifoss.dev/blog/faster-whisper-vs-whispercpp-vs-whisperx-2026/
9. simple-diarizer PyPI: https://pypi.org/project/simple-diarizer/
10. simple-diarizer GitHub: https://github.com/cvqluu/simple_diarizer
11. NVIDIA NeMo Diarization docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
12. diarize GitHub: https://github.com/FoxNoseTech/diarize
13. diarize PyPI: https://pypi.org/project/diarize/
14. diarize API docs: https://foxnosetech.github.io/diarize/api/
15. diarize How It Works: https://foxnosetech.github.io/diarize/how-it-works/
16. pyannote merge diarization+ASR tutorial: https://docs.pyannote.ai/tutorials/diarization-asr-merge
17. WhisperX diarization merge (assign_word_speakers): https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py
18. Combining Whisper + pyannote tutorial: https://theneuralbase.com/whisper/learn/intermediate/combining-whisper-pyannote/
19. Whisper+pyannote subtitle generation guide: https://jamongx.com/whisper-pyannote-speaker-labeled-subtitles-ubuntu-24-04/
20. Speaker Diarization Frameworks comparison: https://dev.to/khushi_nakra_eb3cba0ef3b5/speaker-diarization-frameworks-in-python-tutorial-and-code-walkthrough-n6j
