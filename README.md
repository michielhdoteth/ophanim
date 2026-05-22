# Ophanim

**Local visual perception layer for agents.** Analyze videos, transcribe audio, segment objects, and generate YouTube Shorts -- all locally on your GPU.

```
ophanim probe video.mp4         # Extract metadata
ophanim observe video.mp4        # Analyze video + generate timeline
ophanim ask video.mp4 "..."      # Targeted question answering
ophanim segment video.mp4 "..."  # Segment objects by text prompt
ophanim track video.mp4 "..."    # Track objects across frames
ophanim transcribe video.mp4     # Speech-to-text with Whisper
ophanim status                   # GPU state + cached runs
```

## Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (RTX 4050, 3060, etc.)
- **Python:** 3.12+
- **LM Studio** with a vision model loaded (e.g., `google/gemma-4-e2b`)
- **ffmpeg** (for audio extraction)

## Quick Start

```bash
# Install
pip install -e .
pip install faster-whisper    # For audio transcription
pip install ultralytics       # For SAM segmentation

# Start LM Studio, load a vision model, start the API server
# Default: http://localhost:1234/v1

# Check everything works
ophanim status

# Analyze a video
ophanim observe video.mp4 --json

# With audio transcription
ophanim observe video.mp4 --transcribe --json

# Probe metadata only
ophanim probe video.mp4
```

## Configuration

Default config at `ophanim/config/default.yaml`. Override with env var:

```bash
set OPHANIM_CONFIG=my_config.yaml
```

### Processing Modes

| Mode | Resolution | FPS | Max Frames | Use Case |
|------|-----------|-----|-----------|----------|
| `fast` | 512px | 0.25 | 30 | Quick inspection |
| `balanced` (default) | 768px | 0.5 | 60 | Most workflows |
| `detailed` | 1024px | 1.0 | 180 | High-value clips |

## Architecture

```
ophanim/
  cli/          # Typer CLI (8 commands)
  core/         # Video processing, sampling, GPU management
  providers/    # LM Studio VLM, SAM segmentation, Whisper STT
  storage/      # Config loading, run caching
  config/       # YAML configuration
  tests/        # 140+ tests
```

## Example: YouTube Shorts Pipeline

Ophanim can drive a complete YouTube Shorts production pipeline:

```bash
# 1. Analyze source video
ophanim observe source.mp4 --transcribe --save-memory

# 2. Write script from analysis
# 3. Generate voiceover with Edge TTS
edge-tts --voice en-US-AndrewNeural --rate=+0% --write-media vo.mp3 --text "..."

# 4. Create Manim animation matching voiceover timing
manim -qh scene.py MyScene -o output.mp4

# 5. Compose final Short
ffmpeg -i output.mp4 -i vo.mp3 -filter_complex "[1:a]adelay=500[a]" -map 0:v -map "[a]" -c:v copy final.mp4
```

## License

MIT - see [LICENSE](LICENSE)
