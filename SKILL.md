# openvision -- Video Perception CLI for AI Agents

Open Vision gives AI agents eyes and ears. It extracts keyframes, transcribes audio, and builds structured timelines from video files. Everything runs locally -- no cloud, no telemetry, no API keys required.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `openvision probe <video>` | Video metadata (duration, fps, codec, resolution) |
| `openvision observe <video>` | Visual analysis + cross-modal timeline |
| `openvision ask <video> "question"` | Targeted visual Q&A |
| `openvision transcribe <video>` | Speech-to-text (Parakeet or Whisper) |
| `openvision segment <video> "prompt"` | Text-prompt object segmentation (SAM) |
| `openvision track <video> "prompt"` | Object tracking across frames |
| `openvision status` | GPU, VRAM, provider health, cache stats |
| `openvision status --doctor` | Full diagnostic checks |
| `openvision install parakeet` | Install Parakeet STT model (~350MB) |
| `openvision install whisper --model base` | Install Whisper model |
| `openvision install list-models` | Show all models and install status |
| `openvision observations list` | Saved observation ledgers |
| `openvision observations view <id>` | View a saved observation |

## Global Flags

| Flag | Purpose |
|------|---------|
| `--json` | Output structured JSON (for agent consumption) |
| `--jsonl` | Streaming JSONL output -- one event per line |
| `--provider <name>` | VLM backend: `auto`, `lmstudio`, `ollama`, `llamacpp`, `openai`, `groq`, `together` |
| `--stt-provider <name>` | STT backend: `parakeet` (default), `whisper` |
| `--mode <mode>` | Processing mode: `fast`, `balanced`, `detailed` |
| `--from <time>` | Start time (MM:SS, HH:MM:SS, or seconds) |
| `--to <time>` | End time |
| `--transcribe` / `-t` | Transcribe audio |
| `--diarize` | Add speaker labels (requires pyannote.audio) |
| `--raw-frames` | Skip VLM -- return frame paths for vision-capable agents |
| `--grid` | Generate 3x3 contact sheet |
| `--grid-size <n>` | Grid dimensions (default 3) |
| `--text-anchors` | Force frames at subtitle-cue timestamps |
| `--viewer` | Generate self-contained HTML viewer |
| `--report` | Generate keep/drop visualization HTML |
| `--keep-audio` | Save full soundtrack as audio.m4a |
| `--cookies <file>` | Netscape cookie file for authenticated videos |
| `--cookies-from-browser <name>` | Extract cookies from browser (chrome, firefox, etc.) |
| `--dnn-model <path>` | ONNX model for inline DNN inference during extraction |
| `--save-observations` | Save observation ledger to ~/.openvision/observations/ |

## Backend Selection

### VLM Backends (vision analysis)

Auto-detection order: LM Studio -> Ollama -> llama.cpp -> cloud APIs

| Backend | Endpoint | Notes |
|---------|----------|-------|
| LM Studio | `localhost:1234/v1` | Default local |
| Ollama | `localhost:11434` | Native API |
| llama.cpp | `localhost:8080/v1` | Direct GGUF |
| OpenAI | `api.openai.com/v1` | GPT-4V, GPT-4o |
| Groq | `api.groq.com/openai/v1` | Fast cloud |
| Together AI | `api.together.xyz/v1` | Open-source VLMs |

### STT Backends (speech-to-text)

| Backend | Default | Notes |
|---------|---------|-------|
| Parakeet TDT 0.6B v3 | Yes | INT8 quantized via sherpa-onnx, ~640MB, CPU |
| Whisper | No | Requires `faster-whisper` package |

## Workflows

### Basic video analysis

```bash
openvision observe video.mp4
openvision observe video.mp4 --json
```

### Vision-capable agent (recommended)

```bash
openvision observe video.mp4 --raw-frames --transcribe --json
```

Returns frame paths + audio timeline. Your agent reads the frames directly.

### Text-only agent

```bash
openvision observe video.mp4 --json
```

Open Vision runs a local VLM and returns structured observations.

### Targeted Q&A

```bash
openvision ask video.mp4 "what color is the car?"
openvision ask video.mp4 "is there a dog in this video?" --json
```

### Transcription

```bash
openvision transcribe video.mp4
openvision transcribe video.mp4 --diarize
openvision transcribe video.mp4 --from 1:30 --to 2:00
```

### Contact sheet

```bash
openvision observe video.mp4 --grid
openvision observe video.mp4 --grid --grid-size 4
```

### HTML viewer

```bash
openvision observe video.mp4 --viewer --transcribe
```

Opens in any browser -- no server needed.

### Streaming for pipelines

```bash
openvision observe video.mp4 --jsonl --transcribe
```

Each event is a single JSON line: start, probe, frame, transcript, summary, done.

### YouTube videos

```bash
openvision observe "https://youtube.com/watch?v=..."
openvision observe "https://youtube.com/watch?v=..." --cookies-from-browser chrome
```

### Cross-video search

```bash
# After processing multiple videos, results cached automatically
openvision observe video1.mp4 --json
openvision observe video2.mp4 --json
```

### Diagnose issues

```bash
openvision status --doctor
```

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

## Data Paths

| Path | Default |
|------|---------|
| Home | `~/.openvision` (`OPENVISION_HOME` env var) |
| Observations | `~/.openvision/observations/videos` |
| Downloads | `~/.openvision/downloads` |
| Runs (SQLite) | `~/.openvision/runs/runs.db` |

## Safety

1. **Provider must be running.** Start your VLM backend first (LM Studio, Ollama, etc.) or use `--raw-frames` to skip the VLM.
2. **Parakeet model downloads on first use.** ~640MB cached in `~/.openvision/models/`.
3. **YouTube cookies.** Some videos require authentication. Use `--cookies-from-browser chrome`.
4. **`--jsonl` is streaming.** Events emit as they happen. Pipe to `jq` for pretty-printing.
5. **`--raw-frames` skips the VLM.** Fastest mode for vision-capable agents.
6. **Time windowing is inclusive.** `--from 1:30 --to 2:00` includes both endpoints.
7. **SQLite cache auto-imports.** Legacy JSON runs migrate on first access.
