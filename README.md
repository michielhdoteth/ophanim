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
openvision status                   # GPU, VRAM, cache, data paths
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

## Why CLI-first?

Most local agents can already call shell commands. Open Vision uses that path directly:

```bash
openvision observe ./video.mp4 --json
openvision ask ./video.mp4 "What happens after the person enters?" --json
openvision transcribe ./video.mp4 --json
```

No server required. No cloud upload. No MCP ceremony. Just executable capability for shell-capable agents.

## Supported Backends

| Backend | Status | Notes |
|---------|--------|-------|
| LM Studio | ✅ Full | OpenAI-compatible API |
| Ollama | 🚧 Planned | Local inference |
| llama.cpp | 🚧 Planned | Direct GGUF loading |
| OpenAI API | 🚧 Planned | GPT-4V, etc. |

## Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (RTX 4050, 3060, etc.) — recommended but not required
- **Python:** 3.12+
- **LM Studio** with a vision model loaded (e.g., `google/gemma-4-e2b`)
- **ffmpeg** (for audio extraction)

## Quick Start

```bash
pip install -e .
pip install faster-whisper    # Audio transcription
pip install ultralytics       # SAM segmentation (optional)

# Start LM Studio, load vision model, start API server (default: localhost:1234/v1)

openvision status                 # Verify everything
openvision probe video.mp4        # Inspect video
openvision observe video.mp4 --json    # Analyze
openvision observe video.mp4 --transcribe --json  # With audio
```

## Configuration

Default: `openvision/config/default.yaml`. Override:

```bash
set OPENVISION_CONFIG=my_config.yaml
```

### Processing Modes

| Mode | Resolution | FPS | Max Frames | Use Case |
|------|-----------|-----|-----------|----------|
| `fast` | 512px | 0.25 | 30 | Quick inspection |
| `balanced` (default) | 768px | 0.5 | 60 | Most workflows |
| `detailed` | 1024px | 1.0 | 180 | High-value clips |

## Architecture

```
openvision/
  cli/          # Typer CLI (8 commands)
  core/         # Video, GPU, sampling, audio, errors
  providers/    # LM Studio VLM, SAM, Whisper
  storage/      # Config, cache
  tests/        # 140+ tests
  config/       # default.yaml
```

## Status

Open Vision is a self-hosted AI vision tool for local agent workflows.

- CLI: available
- Video/image processing: available
- Transcription (faster-whisper): available
- SAM segmentation: on-demand
- Multi-provider support: in progress
- Agent memory tools: available
- MCP adapter: planned
- HTTP API: planned

## Roadmap

- [ ] Multi-provider architecture (Ollama, llama.cpp, OpenAI-generic)
- [ ] Auto-detection of running providers
- [ ] URL image fetching
- [ ] Batch processing
- [ ] Screen capture
- [ ] Setup wizard
- [ ] Model download helper
- [ ] HTTP API server
- [ ] MCP adapter

## Maintenance

This repo is released as-is. It is open source because other builders may find it useful, but it is not a heavily maintained product.

If it works for your setup, great. If not, fork it, adapt it, or open an issue with enough detail to reproduce the problem.

## License

MIT
