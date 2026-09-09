# Changelog

All notable changes to Open Vision will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-08

### Added

- **Parakeet TDT 0.6B v3** as default STT provider (INT8 quantized via sherpa-onnx, ~640MB, 10x faster than Whisper on CPU)
- **Provider selection** for STT: `--provider parakeet|whisper` on transcribe and observe commands
- **Python API**: `from openvision import process, transcribe` for programmatic access
- **Streaming JSONL output**: `--jsonl` flag on observe, transcribe, probe, segment, ask commands
- **SQLite run storage**: Replaced flat JSON with SQLite database (`runs.db`) for run metadata, artifacts, frames, and transcripts
- **Cross-video search index**: `core.memory.VideoMemory` for searching frames and transcripts across all processed videos
- **Contact sheets**: `--grid` flag generates 3x3 keyframe grids
- **Text anchors**: `--text-anchors` forces frame extraction at subtitle-cue timestamps (parses .srt/.vtt)
- **Local HTML viewer**: `--viewer` generates self-contained HTML with base64-embedded keyframes and transcript
- **Keep/drop report**: `--report` visualizes frame selection decisions as HTML timeline
- **Time windowing**: `--from` and `--to` flags for processing video segments
- **Cookie support**: `--cookies` and `--cookies-from-browser` for authenticated video downloads
- **Keep audio**: `--keep-audio` saves full soundtrack as audio.m4a
- **OpenCV 5 DNN provider**: `providers.opencv_dnn.OpenCVDNNProvider` for running ONNX models via OpenCV 5's rewritten DNN engine (ENGINE_AUTO/NEW/CLASSIC)
- **FFmpeg ONNX Runtime DNN filter**: `core.dnn_filter.DNNFilterPipeline` for inline ONNX model inference during frame extraction, with `--dnn-model` flag on observe
- **FFmpeg 9.0 CUDA transpose**: GPU-accelerated frame orientation correction with automatic CPU fallback
- **OpenCV 5 upgrade**: `pyproject.toml` bumped to `opencv-python>=5.0`
- **yt-dlp upgrade**: Updated to 2026.8.19 for YouTube download support
- **Doctor diagnostics**: `openvision status --doctor` checks ffmpeg, yt-dlp, GPU, OpenCV 5 engine, providers, parakeet model, and Python packages
- **155 new unit tests** covering stream, memory, contact_sheet, report, viewer, parakeet, opencv_dnn, dnn_filter, cache (SQLite), and api modules

### Changed

- `whisper_device` renamed to `stt_device` throughout the codebase
- `storage.cache.RunCache` rewritten to use SQLite with backward-compatible auto-import of existing JSON runs
- All commands now use `--provider` / `--stt-provider` flags instead of hardcoded whisper references
- README rewritten to be product-focused (less technical, more use-case driven)

### Removed

- All external project references (Agent-Reach)
- MCP adapter (not planned -- CLI access is cleaner)

## [0.9.0] - 2026-07-30

### Added

- Initial release with 10 commands (observe, ask, transcribe, segment, track, probe, ground, status, observations)
- Multi-provider VLM architecture (LM Studio, Ollama, llama.cpp, OpenAI, Groq, Together AI, vLLM)
- Auto-detection of running backends
- Processing modes (fast, balanced, detailed)
- Cross-modal timeline (visual + audio)
- Raw frames mode for vision-capable agents
- Speaker diarization support (pyannote.audio)
- SAM segmentation (ultralytics)
- Object tracking
- Observation ledgers
