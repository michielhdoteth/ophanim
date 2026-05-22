# Ophanim

A local visual perception layer for agents. Process videos and images using LM Studio (Gemma 4 E2B) for vision-language inference, with optional SAM segmentation.

## Requirements

- Python 3.9+
- NVIDIA GPU with 6GB+ VRAM (RTX 4050, etc.)
- [LM Studio](https://lmstudio.ai/) with a vision model loaded (e.g., Gemma 4 E2B)
- CUDA-capable PyTorch (included with install)

## Installation

```bash
# From project root
pip install -e .

# Optional: SAM segmentation support
pip install transformers ultralytics
```

## Quick Start

1. **Start LM Studio** and load `google/gemma-4-e2b` (or any vision model)
2. **Verify LM Studio is running** at `http://localhost:1234/v1`

```bash
# Check GPU and system status
ophanim status

# Probe a video for metadata
ophanim probe video.mp4

# Observe a video (smart sampling + VLM analysis)
ophanim observe video.mp4

# Ask a specific question
ophanim ask video.mp4 "Is there a red car?"

# Segment an object (requires SAM deps)
ophanim segment video.mp4 "person" --start 5 --end 30

# Track an object across frames
ophanim track video.mp4 "car"
```

## Commands

| Command | Description |
|---------|-------------|
| `probe` | Extract video metadata without full processing |
| `observe` | Analyze video/image and return timeline + summary |
| `ask` | Ask a specific question about video content |
| `segment` | Segment objects in video by text prompt |
| `track` | Track object positions across frames |
| `status` | Show GPU state, VRAM, cached runs, config |
| `memory` | List/view/delete saved observations |

## Configuration

Config file: `ophanim/config/default.yaml`

Override with `OPHANIM_CONFIG` environment variable:

```bash
set OPHANIM_CONFIG=path/to/your/config.yaml
ophanim status
```

### Processing Modes

| Mode | Resolution | FPS | Max Frames | Use Case |
|------|-----------|-----|-----------|----------|
| `fast` | 512px | 0.25 | 30 | Quick inspection, long videos |
| `balanced` (default) | 768px | 0.5 | 60 | Most agent workflows |
| `detailed` | 1024px | 1.0 | 180 | High-value analysis, short clips |

## Output Options

- **Human-readable**: Rich-formatted terminal output (default)
- **Machine-readable**: `--json` flag on all commands
- **Persistent memory**: `--save-memory` on observe saves to `memory/videos/`

## Caching

Results are cached in `runs/` directory. Cached results are reused automatically. Use `--force` to reprocess.

```bash
ophanim observe video.mp4          # Uses cache
ophanim observe video.mp4 --force  # Reprocesses
```

## Error Handling

All commands return recoverable errors with suggested retry parameters:

```json
{
  "error": "GPU_OUT_OF_MEMORY",
  "message": "...",
  "suggested_retry": {"mode": "fast", "max_resolution": 512}
}
```

## Architecture

```
ophanim/
  cli/          # Typer CLI commands
  core/         # Video processing, sampling, GPU management
  providers/    # LM Studio VLM, SAM segmentation
  storage/      # Config loading, run caching
  config/       # YAML configuration
```

## License

MIT
