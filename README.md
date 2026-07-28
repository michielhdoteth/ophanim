# Open Vision

**Self-hosted, privacy-first AI vision tool for CLI and agents.**

Open Vision is a local CLI that gives shell-capable agents visual perception: video inspection, frame sampling, visual Q&A, transcription, segmentation, tracking, and compact JSON/Markdown outputs.

Built for local agent workflows — no cloud upload, no telemetry, full privacy.

## Commands

```
openvision --version                # Version + data paths
openvision probe video.mp4          # Video metadata
openvision observe video.mp4        # Visual analysis + timeline
openvision observe "https://..."    # Native yt-dlp download + observe
openvision ask video.mp4 "..."      # Targeted visual question
openvision segment video.mp4 "..."  # Text-prompt segmentation
openvision track video.mp4 "..."    # Object tracking
openvision transcribe video.mp4     # Speech-to-text
openvision status                   # GPU, VRAM, provider health, cache
openvision memory list              # Saved observations (~/.openvision/memory)
```

### Data paths (stable, not CWD-relative)

| Path | Default |
|------|---------|
| Home | `~/.openvision` (`OPENVISION_HOME`) |
| Memory | `~/.openvision/memory/videos` |
| Downloads | `~/.openvision/downloads` |
| Runs | `~/.openvision/runs` |

```bash
openvision observe "https://www.youtube.com/watch?v=ID" --mode balanced --detail efficient -t --save-memory --json
openvision memory list
```

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
pip install ultralytics       # SAM segmentation (optional)

# Start any supported backend (e.g. LM Studio, Ollama, llama.cpp)
# Then:
openvision status              # Verify provider connectivity
openvision probe video.mp4
openvision observe video.mp4 --json
openvision observe video.mp4 --transcribe --json
```

### Provider Selection

```bash
# Auto-detect (default)
openvision observe video.mp4 --provider auto

# Explicit backend
openvision observe video.mp4 --provider lmstudio
openvision observe video.mp4 --provider ollama
openvision observe video.mp4 --provider llamacpp

# Cloud APIs (set API key env var first)
export OPENAI_API_KEY="sk-..."
openvision observe video.mp4 --provider openai

export GROQ_API_KEY="gsk_..."
openvision observe video.mp4 --provider groq

export TOGETHER_API_KEY="..."
openvision observe video.mp4 --provider together
```

### Model Selection

```bash
# Auto-detect loaded model (default)
openvision observe video.mp4 --model auto

# Specify model name
openvision observe video.mp4 --model "llava:13b"
openvision observe video.mp4 --model "gpt-4o"
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

### Segmentation

SAM segmentation isolates objects in extracted frames. Requires `pip install ultralytics`.

```bash
# Standalone segmentation
openvision segment video.mp4 "a person wearing red"

# Inline with observe (on-demand in balanced mode)
openvision observe video.mp4 --segment

# Always segment in detailed mode
openvision observe video.mp4 --mode detailed
```

## Architecture

```
openvision/
  cli/              # Typer CLI (8 commands)
    commands/
      observe.py    # Main analysis + inline segmentation
      ask.py        # Visual Q&A
      segment.py    # Standalone SAM segmentation
      track.py      # Object tracking
      status.py     # Provider health + GPU info
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
  storage/          # Config, cache
  models.py         # Pydantic schemas
  config/           # default.yaml
  tests/            # 250+ tests
```

## Status

- CLI: available
- Video/image processing: available
- Multi-provider VLM support: available (7 backends)
- Auto-detection: available
- Transcription (faster-whisper): available
- SAM segmentation: available (optional dependency)
- Object tracking: available
- Agent memory tools: available
- Mode-based resource control: available
- MCP adapter: planned
- HTTP API: planned

## Roadmap

- [x] Multi-provider architecture (LM Studio, Ollama, llama.cpp, OpenAI, Groq, Together)
- [x] Auto-detection of running providers
- [x] Mode-based segmentation control
- [ ] URL image fetching
- [ ] Batch processing
- [ ] Screen capture
- [ ] Setup wizard
- [ ] Model download helper
- [ ] OpenCV 5 DNN integration (YOLO26, SAM 2)
- [ ] FFmpeg 8.x vfrdet for variable frame rate detection
- [ ] HTTP API server
- [ ] MCP adapter

## Maintenance

This repo is released as-is. It is open source because other builders may find it useful, but it is not a heavily maintained product.

If it works for your setup, great. If not, fork it, adapt it, or open an issue with enough detail to reproduce the problem.

## License

MIT
