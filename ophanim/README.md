# Ophanim

**Local vision CLI for agents.**

Ophanim is a local CLI that gives shell-capable agents basic visual perception: video inspection, frame sampling, visual Q&A, transcription, segmentation, tracking, and compact JSON/Markdown outputs.

I built it for my own local agent workflows and open-sourced it in case it helps other people building local agents.

## Commands

```
ophanim probe video.mp4         # Video metadata
ophanim observe video.mp4        # Visual analysis + timeline
ophanim ask video.mp4 "..."      # Targeted visual question
ophanim segment video.mp4 "..."  # Text-prompt segmentation
ophanim track video.mp4 "..."    # Object tracking
ophanim transcribe video.mp4     # Speech-to-text
ophanim status                   # GPU, VRAM, cache
ophanim memory list              # Saved observations
```

## Why CLI-first?

Most local agents can already call shell commands. Ophanim uses that path directly:

```bash
ophanim observe ./video.mp4 --json
ophanim ask ./video.mp4 "What happens after the person enters?" --json
ophanim transcribe ./video.mp4 --json
```

No server required. No cloud upload. No MCP ceremony. Just executable capability for shell-capable agents.

MCP may be added later as a thin adapter, but the CLI is the core interface.

## Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (RTX 4050, 3060, etc.)
- **Python:** 3.12+
- **LM Studio** with a vision model loaded (e.g., `google/gemma-4-e2b`)
- **ffmpeg** (for audio extraction)

## Quick Start

```bash
pip install -e .
pip install faster-whisper    # Audio transcription
pip install ultralytics       # SAM segmentation (optional)

# Start LM Studio, load vision model, start API server (default: localhost:1234/v1)

ophanim status                 # Verify everything
ophanim probe video.mp4        # Inspect video
ophanim observe video.mp4 --json    # Analyze
ophanim observe video.mp4 --transcribe --json  # With audio
```

## Configuration

Default: `ophanim/config/default.yaml`. Override:

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
  core/         # Video, GPU, sampling, audio, errors
  providers/    # LM Studio VLM, SAM, Whisper
  storage/      # Config, cache
  tests/        # 140+ tests
  config/       # default.yaml
```

## Status

Ophanim is a personal tool I built for my own local agent workflows. It is open source because other builders may find it useful, but it is not a heavily maintained product.

- CLI: available
- Video/image processing: available
- Transcription (faster-whisper): available
- SAM segmentation: on-demand
- Content workflows: available
- MCP adapter: planned
- Agent memory tools: planned

## Roadmap

- [ ] Better GPU-safe scheduling
- [ ] Scene-change guided frame sampling
- [ ] JSON schema stabilization
- [ ] OpenClaw/opencode skill examples
- [ ] Squish memory export
- [ ] Optional MCP adapter

## Maintenance

This repo is released as-is. I may update it when I need new functionality for my own agent stack. Contributions are welcome, but there is no guaranteed support timeline.

If it works for your setup, great. If not, fork it, adapt it, or open an issue with enough detail to reproduce the problem.

## License

MIT
