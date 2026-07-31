# Open Vision

**Agents are blind by default. Open Vision gives them eyes and ears.**

A local CLI that gives shell-capable agents visual perception. No cloud, no telemetry, no API keys required.

```
openvision observe video.mp4          # See what's in a video
openvision ask video.mp4 "what color is the car?"  # Targeted visual Q&A
openvision transcribe video.mp4       # Speech-to-text with speaker labels
openvision --version                  # Version + data paths
```

Open Vision is the eyes for agent frameworks — read the full [Open Vision Agent Guide](https://github.com/michielhdoteth/openvision/blob/main/AGENT_GUIDE.md) to get started.

## Two Operating Modes

### Agent with vision (recommended)

If your agent supports image input (Claude Code, etc.), use `--raw-frames`:

```bash
openvision observe video.mp4 --raw-frames --transcribe --diarize --json
```

This skips the local VLM entirely — just extracts frames, runs audio transcription, and returns frame paths + audio timeline. Your vision-capable agent reads the frames directly.

### Agent without vision

Use the default mode with a local VLM backend (LM Studio, Ollama, llama.cpp):

```bash
openvision observe video.mp4 --json
```

Open Vision analyzes each frame locally and returns structured observations.

## Commands

```
openvision --version                  # Version + data paths
openvision probe video.mp4            # Video metadata (duration, fps, codec)
openvision observe video.mp4          # Visual analysis + cross-modal timeline
openvision observe "https://..."      # Native yt-dlp download + observe
openvision ask video.mp4 "..."        # Targeted visual question
openvision segment video.mp4 "..."    # Text-prompt segmentation (SAM)
openvision track video.mp4 "..."      # Object tracking
openvision transcribe video.mp4       # Speech-to-text (faster-whisper)
openvision status                     # GPU, VRAM, provider health, cache
openvision observations list          # Saved observation ledgers
```

### Key Flags

| Flag | Effect |
|------|--------|
| `--raw-frames` | Skip VLM — return frame paths + audio timeline for vision-capable agents |
| `--transcribe` / `-t` | Transcribe audio with faster-whisper |
| `--diarize` | Add speaker labels to transcription (requires `pyannote.audio`) |
| `--save-observations` | Save a machine-readable observation ledger to `~/.openvision/observations/` |
| `--json` | Output structured JSON (for agent consumption) |
| `--provider` | Select VLM backend: `auto`, `lmstudio`, `ollama`, `llamacpp`, `openai`, `groq`, `together` |
| `--mode` | Processing mode: `fast`, `balanced`, `detailed` |
| `--segment` / `-s` | Run SAM segmentation on extracted frames |

### Cross-Modal Timeline

When `--transcribe` or `--diarize` is used, the timeline includes audio segments aligned with visual observations:

```json
{
  "timeline": [
    {
      "time_seconds": 0.0,
      "timestamp": "00:00",
      "observation": "Speaker says: welcome to the presentation",
      "speaker": "SPEAKER_00",
      "modality": "audio"
    },
    {
      "time_seconds": 5.2,
      "timestamp": "00:05",
      "observation": "A person in a blue shirt is standing at a podium",
      "frame_path": "/path/to/frame_0002.jpg",
      "modality": "visual"
    }
  ]
}
```

Fields:
- **`speaker`**: Speaker ID from diarization (e.g. `SPEAKER_00`), `null` for visual entries
- **`modality`**: `"visual"`, `"audio"`, or `"segmentation"` — lets agents filter timeline by data type

### Data Paths (stable, not CWD-relative)

| Path | Default |
|------|---------|
| Home | `~/.openvision` (`OPENVISION_HOME`) |
| Observations | `~/.openvision/observations/videos` |
| Downloads | `~/.openvision/downloads` |
| Runs | `~/.openvision/runs` |

### Observation Ledgers

`--save-observations` writes a structured markdown file combining summary + timeline + transcript:

```bash
openvision observe video.mp4 --transcribe --diarize --save-observations
openvision observations list
openvision observations view 2026-07-30-my-video
```

This is a machine-readable ledger of the whole video — no need to rewatch or re-analyze.

## Supported Backends

| Backend | Status | Endpoint | Notes |
|---------|--------|----------|-------|
| LM Studio | ✅ Full | `localhost:1234/v1` | Default local backend |
| Ollama | ✅ Full | `localhost:11434` | Native `/api/chat` |
| llama.cpp | ✅ Full | `localhost:8080/v1` | Direct GGUF loading |
| OpenAI API | ✅ Full | `api.openai.com/v1` | GPT-4V, GPT-4o |
| Groq | ✅ Full | `api.groq.com/openai/v1` | Fast cloud inference |
| Together AI | ✅ Full | `api.together.xyz/v1` | Open-source VLMs |
| vLLM | ✅ Full | Custom endpoint | OpenAI-compatible |

**Auto-detection:** Set `provider: "auto"` (default) and Open Vision scans for running backends in order: LM Studio → Ollama → llama.cpp → cloud.

## Quick Start

```bash
pip install -e .

# Optional dependencies
pip install faster-whisper    # Audio transcription
pip install pyannote.audio    # Speaker diarization
pip install ultralytics       # SAM segmentation

# Start any supported backend (e.g. LM Studio, Ollama, llama.cpp)
# Then:
openvision status              # Verify provider connectivity
openvision probe video.mp4
openvision observe video.mp4 --json
openvision observe video.mp4 --transcribe --diarize --json
```

### Provider Selection

```bash
# Auto-detect (default)
openvision observe video.mp4 --provider auto

# Explicit backend
openvision observe video.mp4 --provider lmstudio
openvision observe video.mp4 --provider ollama

# Cloud APIs (set API key env var first)
export OPENAI_API_KEY="sk-..."
openvision observe video.mp4 --provider openai

export GROQ_API_KEY="gsk_..."
openvision observe video.mp4 --provider groq
```

## Configuration

Default: `openvision/config/default.yaml`. Override:

```bash
set OPENVISION_CONFIG=my_config.yaml
```

### Processing Modes

| Mode | Resolution | FPS | Max Frames | Segmentation | Use Case |
|------|-----------|-----|-----------|-------------|----------|
| `fast` | 512px | 0.25 | 30 | Off | Quick inspection |
| `balanced` (default) | 768px | 0.5 | 60 | On-demand (`--segment`) | Most workflows |
| `detailed` | 1024px | 1.0 | 180 | Always on | High-value clips |

## Architecture

```
openvision/
  cli/              # Typer CLI (8 commands)
    commands/
      observe.py    # Main analysis + raw-frames mode
      ask.py        # Visual Q&A + raw-frames mode
      observations.py  # Saved observation ledgers
      segment.py    # Standalone SAM segmentation
      track.py      # Object tracking
      status.py     # Provider health + GPU info
      transcribe.py # Speech-to-text
  core/             # Video, GPU, sampling, audio, errors
  providers/        # Multi-provider VLM architecture
    base.py         # Abstract VlmProvider interface
    openai_compat.py # Shared OpenAI-compatible base
    lmstudio.py     # LM Studio provider
    ollama.py       # Ollama native provider
    llamacpp.py     # llama.cpp provider
    cloud.py        # OpenAI/Groq/vLLM/Together/etc.
    registry.py     # Auto-detection + factory
    sam.py          # SAM segmentation provider
    whisper.py      # Whisper transcription provider
  storage/          # Config, cache, paths
  models.py         # Pydantic schemas (TimelineEntry, RawFrame, etc.)
  config/           # default.yaml
  tests/            # 250+ tests
```

## Status

- CLI: available
- Video/image processing: available
- Multi-provider VLM support: available (7 backends)
- Auto-detection: available
- Transcription (faster-whisper): available
- Speaker diarization (pyannote): available
- SAM segmentation: available (optional dependency)
- Object tracking: available
- Raw frames mode (vision-capable agents): available
- Cross-modal timeline (visual + audio): available
- Mode-based resource control: available
- HTTP API: planned
- MCP adapter: not planned (CLI access is cleaner)

## License

MIT
