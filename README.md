# Open Vision

**Agents are blind by default. Open Vision gives them eyes and ears.**

A local CLI that lets AI agents see and hear videos. No cloud, no API keys, no telemetry. Everything runs on your machine.

```
openvision observe video.mp4
openvision ask video.mp4 "what color is the car?"
openvision transcribe video.mp4
```

## Why?

AI agents can read code, but they can't watch a video, read a screen recording, or hear a meeting. Open Vision fixes that. It extracts keyframes, transcribes audio, and builds a structured timeline that agents can actually reason about.

Works with any agent framework -- Claude Code, LangChain, CrewAI, or just a shell script.

## Install

```bash
pip install -e .
```

That's it. yt-dlp is included for YouTube URLs. Parakeet STT is bundled for speech-to-text. No GPU required (but it's faster if you have one).

**Optional extras:**

```bash
pip install "openvision[diarize]"    # Speaker labels
pip install "openvision[segment]"    # Object segmentation (SAM)
```

## 30-Second Start

```bash
# Check that everything works
openvision status

# Analyze a video
openvision observe video.mp4

# Ask a specific question
openvision ask video.mp4 "is there a dog in this video?"

# Transcribe with speaker labels
openvision transcribe video.mp4 --diarize

# Works with YouTube too
openvision observe "https://youtube.com/watch?v=..."
```

## What It Does

**Observe** -- Extracts keyframes from video, analyzes each with a vision model, and builds a cross-modal timeline (visual + audio). Returns a structured summary.

**Ask** -- Targeted question about video content. "What color is the car?" returns a direct answer with evidence frames.

**Transcribe** -- Speech-to-text using Parakeet (default, fast, runs on CPU) or Whisper. Supports speaker diarization.

**Segment** -- Text-prompt object segmentation. "person" returns masked regions with bounding boxes.

**Track** -- Frame-by-frame object tracking across a video.

**Probe** -- Video metadata: duration, resolution, FPS, codec, estimated processing cost.

## For Agents

### Vision-capable agents (recommended)

If your agent can read images (Claude Code, GPT-4V, etc.):

```bash
openvision observe video.mp4 --raw-frames --transcribe --json
```

This skips the local VLM and returns frame paths + audio timeline. Your agent does the visual reasoning.

### Text-only agents

```bash
openvision observe video.mp4 --json
```

Open Vision runs a local VLM (LM Studio, Ollama, llama.cpp) and returns structured observations.

### Streaming pipelines

```bash
openvision observe video.mp4 --jsonl --transcribe
```

One JSON event per line. Agents and pipelines can consume results incrementally.

## Features

| Feature | Command |
|---------|---------|
| Visual analysis | `observe video.mp4` |
| Targeted Q&A | `ask video.mp4 "question"` |
| Speech-to-text | `transcribe video.mp4` |
| Speaker labels | `transcribe video.mp4 --diarize` |
| Object segmentation | `segment video.mp4 "prompt"` |
| Object tracking | `track video.mp4 "prompt"` |
| YouTube support | `observe "https://..."` |
| Time windowing | `observe video.mp4 --from 1:30 --to 2:00` |
| Contact sheets | `observe video.mp4 --grid` |
| HTML viewer | `observe video.mp4 --viewer` |
| Frame selection report | `observe video.mp4 --report` |
| Keep audio | `observe video.mp4 --keep-audio` |
| Cookie auth | `observe "url" --cookies-from-browser chrome` |
| Inline DNN inference | `observe video.mp4 --dnn-model yolov8n.onnx` |
| Streaming JSONL | `observe video.mp4 --jsonl` |
| Python API | `from openvision import process, transcribe` |

## Python API

```python
from openvision import process, transcribe

result = process("video.mp4", mode="balanced")
print(result.summary)
print(result.timeline)

transcript = transcribe("video.mp4")
for seg in transcript.segments:
    print(f"[{seg.start:.1f}s] {seg.text}")
```

## Backends

Open Vision auto-detects what's running. No config needed.

**Vision models:** LM Studio, Ollama, llama.cpp, OpenAI, Groq, Together AI, vLLM

**Speech-to-text:** Parakeet TDT 0.6B v3 (default, INT8, ~640MB, CPU), Whisper (optional)

**Hardware:** Works on CPU. GPU (CUDA) is faster for VLM inference. Parakeet runs efficiently on CPU.

## Configuration

Defaults work out of the box. Customize via `config/default.yaml` or environment variables:

```bash
export OPENVISION_CONFIG=my_config.yaml
```

Three processing modes:

| Mode | Best for |
|------|----------|
| `fast` | Quick inspection, low resource usage |
| `balanced` (default) | Most workflows |
| `detailed` | High-value clips, maximum detail |

All data lives in `~/.openvision` (override with `OPENVISION_HOME`).

## Contributing

```bash
git clone https://github.com/michielhdoteth/openvision.git
cd openvision
pip install -e ".[test]"
pytest tests/ -v
```

Architecture: `cli/` (Typer commands) -> `core/` (video, audio, sampling, streaming) -> `providers/` (VLM + STT backends) -> `storage/` (SQLite cache, paths). See `AGENT_GUIDE.md` for the full agent integration guide.

## License

MIT
