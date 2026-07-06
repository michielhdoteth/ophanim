# Ophanim Restructure + Improvement Plan

**Date:** 2026-07-06
**Status:** DRAFT - AWAITING APPROVAL

---

## Overview

Three work streams executed sequentially:
1. Restructure to `src/ophanim/` layout (eliminate ophanim/ophanim/ nesting)
2. Fix transcription pipeline (timeout, GPU, flags)
3. Cannibalize useful patterns from claude-video repo

---

## Phase 1: Restructure to src/ophanim/ Layout

### Goal
Eliminate the confusing `ophanim/ophanim/` nesting. Standard Python `src/` layout.

### Before
```
ophanim/                    <- repo root
├── __init__.py             <- STALE duplicate
├── __main__.py             <- duplicate
├── cli/                    <- STALE duplicate (root has old code)
├── config/                 <- duplicate
├── core/                   <- STALE duplicate
├── models.py               <- STALE (missing TokenUsage)
├── providers/              <- STALE duplicate
├── storage/                <- duplicate
├── tests/                  <- duplicate
├── README.md               <- duplicate
├── ophanim/                <- THE PACKAGE (newer code with TokenUsage, VlmResponse)
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli/
│   ├── config/
│   ├── core/
│   ├── models.py           <- NEWER (has TokenUsage class)
│   ├── providers/          <- NEWER (has VlmResponse, TokenUsage)
│   ├── storage/
│   ├── tests/
│   └── README.md
├── .git/
├── .gitignore
├── .opencode/
├── LICENSE
├── ophanim.egg-info/
├── pyproject.toml
└── runs/
```

### After
```
ophanim/                    <- repo root
├── src/
│   └── ophanim/            <- THE PACKAGE (canonical source)
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── app.py
│       │   └── commands/
│       │       ├── ask.py
│       │       ├── memory.py
│       │       ├── observe.py
│       │       ├── probe.py
│       │       ├── segment.py
│       │       ├── status.py
│       │       ├── track.py
│       │       └── transcribe.py
│       ├── config/
│       │   └── default.yaml
│       ├── core/
│       │   ├── audio.py
│       │   ├── errors.py
│       │   ├── gpu.py
│       │   ├── image.py
│       │   ├── sampling.py
│       │   └── video.py
│       ├── models.py
│       ├── providers/
│       │   ├── base.py
│       │   ├── lmstudio.py
│       │   ├── sam.py
│       │   └── whisper.py
│       ├── storage/
│       │   ├── cache.py
│       │   └── config.py
│       └── tests/
│           ├── test_*.py (14 files)
├── .git/
├── .gitignore
├── .opencode/
├── LICENSE
├── pyproject.toml          <- UPDATED for src layout
├── README.md               <- single copy at root
└── runs/
```

### Steps

1. **Create `src/` directory**
   ```bash
   mkdir src
   ```

2. **Move the PACKAGE (newer code) into src/**
   ```bash
   # Move the ophanim/ subfolder (which has the newer code) to src/ophanim/
   mv ophanim/ src/ophanim_temp
   # Now root-level files are exposed. Delete them.
   ```

3. **Delete ALL root-level duplicates**
   ```bash
   rm __init__.py __main__.py models.py README.md
   rm -rf cli/ config/ core/ providers/ storage/ tests/
   rm -rf ophanim.egg-info/
   ```

4. **Rename the moved package into place**
   ```bash
   mv src/ophanim_temp src/ophanim
   ```

5. **Move README.md to root** (keep one copy)
   ```bash
   cp src/ophanim/README.md README.md
   ```

6. **Update pyproject.toml**
   ```toml
   [build-system]
   requires = ["setuptools>=68"]
   build-backend = "setuptools.build_meta"

   [project]
   name = "ophanim"
   version = "0.1.0"
   description = "Local visual perception layer for agents"
   requires-python = ">=3.12"
   dependencies = [
       "opencv-python>=4.13",
       "pillow>=11",
       "numpy>=2",
       "typer>=0.23",
       "rich>=13",
       "pydantic>=2",
       "httpx>=0.28",
       "pyyaml>=6",
       "nvidia-ml-py>=12",
       "imageio>=2.36",
       "scikit-learn>=1.6",
       "scipy>=1.13",
       "tqdm>=4.67",
       "click>=8",
   ]

   [project.scripts]
   ophanim = "ophanim.__main__:main"

   [tool.setuptools.packages.find]
   where = ["src"]
   include = ["ophanim*"]
   ```

7. **Reinstall in dev mode**
   ```bash
   pip install -e .
   ```

8. **Verify imports work**
   ```bash
   python -c "from ophanim.cli.app import app; print('OK')"
   python -c "from ophanim.providers.whisper import WhisperProvider; print('OK')"
   ```

### Key Notes
- The `ophanim/ophanim/` package has NEWER code (TokenUsage, VlmResponse) -- these are the canonical files
- All imports already use `from ophanim.xxx` -- no import changes needed
- The `src/` layout is the Python community best practice -- it prevents accidental imports from the working directory

---

## Phase 2: Fix Transcription Pipeline

### 2A. Increase ffmpeg timeout (300s -> 900s)

**File:** `src/ophanim/core/audio.py`
**Line 73:** Change `timeout=300` to `timeout=900`
**Line 80:** Update error message from "5 min" to "15 min"

**Why 900s (15 min):** User reported 25-30 min videos take ~10 min on CPU with base model. 900s gives 50% headroom.

### 2B. Add GPU option for Whisper transcription

**File:** `src/ophanim/providers/whisper.py`

Currently hardcoded to CPU. Changes:
1. In `__init__`, default `device` to `"auto"` instead of `"cpu"`
2. Add `_resolve_device()` method:
   - `"auto"` -> try CUDA if available AND VRAM > 2GB free, else CPU
   - `"cpu"` -> always CPU
   - `"cuda"` -> always GPU (with int8 or float16)
3. When using GPU, set `compute_type` to `"float16"` instead of `"int8"`
4. Add `unload()` method that also calls `torch.cuda.empty_cache()` when GPU was used
5. Add logging: "Transcribing on GPU (X.X GB VRAM)" or "Transcribing on CPU"

**File:** `src/ophanim/cli/commands/transcribe.py`

Add `--device` flag:
```python
device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda")
```

Update config:
```python
config = {
    "model_size": model,
    "device": device,  # was hardcoded "cpu"
    "compute_type": "int8" if device == "cpu" else "float16",
}
```

**File:** `src/ophanim/cli/commands/observe.py`

Add `--whisper-device` option or read from config. Currently line 278 creates `WhisperProvider()` with no config.

### 2C. Add --prompt flag (entirely custom prompt)

**File:** `src/ophanim/cli/commands/observe.py`

Add new option after line 55:
```python
prompt: str = typer.Option(None, "--prompt", "-p", help="Custom prompt (replaces default frame description)")
```

Current behavior:
- `--question` = "Answer this question about the image: {question}" -- sends each frame with this prefix
- No flag = "Describe what is happening in this frame. Focus on objects, people, actions."

New behavior:
- `--question` = hardcoded question prompt (unchanged)
- `--prompt` = entirely custom prompt, no prefix added
- Both exist independently

Changes in `_handle_video` (line 233-239):
```python
if prompt:
    resp = provider.describe_image(frame["image"], prompt)
elif question:
    resp = provider.describe_image(frame["image"], question)
else:
    resp = provider.describe_image(
        frame["image"],
        "Describe what is happening in this frame. Focus on objects, people, actions. Be concise (1-2 sentences)."
    )
```

Same change in `_handle_image` (line 119).

---

## Phase 3: Cannibalize from claude-video

### What to take (and where it goes)

#### 3A. Lightweight Frame Deduplication -> `src/ophanim/core/sampling.py`

**Source:** claude-video `frames.py` - `dedupe_perceptual()` + `_frame_delta()` + `_thumb_frames()`

**Why:** The current dedup in `sampling.py` uses OpenCV HSV histograms which requires decoding frames in memory. The claude-video approach uses ffmpeg to create 16x16 grayscale thumbnails (pure stdlib, no image libraries), then computes mean absolute pixel difference. It's:
- Faster (one ffmpeg pass, no OpenCV decode)
- Simpler (pure Python math, no numpy/cv2)
- Effective (threshold 2.0 catches static slides, screen recordings)

**How to integrate:**
Add a new function `dedupe_frames_ffmpeg(frame_paths: list[Path], threshold: float = 2.0) -> list[Path]` to `sampling.py`. This runs AFTER frame extraction, operating on saved JPEG files. The existing `deduplicate()` function stays for backward compatibility but the new pipeline uses the lighter approach.

**Key code to port:**
- `_frame_delta()` - pure Python mean absolute difference (trivial, ~5 lines)
- `_thumb_frames()` - ffmpeg downscale to 16x16 grayscale (1 subprocess call)
- `dedupe_perceptual()` - greedy dedup against last-kept frame

#### 3B. FFmpeg Scene-Change Extraction -> `src/ophanim/core/video.py`

**Source:** claude-video `frames.py` - `extract_scene_candidates()`

**Why:** Currently ophanim's scene detection (`detect_scenes()` in sampling.py) extracts ALL frames into memory first, then compares HSV histograms with OpenCV. The claude-video approach uses ffmpeg's built-in `select='gt(scene,T)'` filter to detect and extract scene-change frames in a single pass. This is:
- Much faster (single ffmpeg decode, not decode-then-compare)
- Lower memory (no need to hold all frames in RAM)
- Can use lower threshold (0.20 vs 30.0 HSV) since ffmpeg's scene detection is more robust

**How to integrate:**
Add `extract_scene_frames(video_path, out_dir, threshold=0.20, max_frames=100, start=None, end=None) -> list[dict]` to `video.py`. Returns list of `{path, timestamp, reason}` dicts. This becomes an alternative to the current `smart_sample()` pipeline.

**Key constants to port:**
- `SCENE_THRESHOLD = 0.20` (ffmpeg scene score)
- `SCENE_MIN_FRAMES = 8` (below this, fall back to uniform sampling)

#### 3C. Auto-FPS Budget System -> `src/ophanim/core/sampling.py`

**Source:** claude-video `frames.py` - `auto_fps()` + `auto_fps_focus()`

**Why:** Currently ophanim uses a fixed `fps` parameter (default 0.5). The claude-video approach calculates fps dynamically based on video duration and a frame budget cap. This means:
- Short videos (<30s) get dense sampling (~1 fps)
- Long videos (>10min) get capped at 100 frames (sparse scan warning)
- User-specified ranges get denser budgets (focus mode)

**How to integrate:**
Add `auto_fps(duration_seconds: float, max_frames: int = 60) -> tuple[float, int]` to `sampling.py`. Returns (fps, target_frame_count). The `smart_sample()` function can optionally use this when `fps` is not explicitly provided.

#### 3D. Caption-First Transcription -> `src/ophanim/providers/whisper.py`

**Source:** claude-video `watch.py` flow + `transcribe.py`

**Why:** Currently ophanim always runs local Whisper (slow, 10 min for 30 min video). claude-video tries captions first (yt-dlp pulls them for free), and only falls back to Whisper when no captions exist. For YouTube videos, this is instant and free.

**How to integrate:**
Add a new method to `WhisperProvider`:
```python
def try_captions(self, video_path: str) -> Optional[Transcript]:
    """Try to extract captions via yt-dlp. Returns None if unavailable."""
```

This:
1. Checks if `yt-dlp` is installed
2. Runs `yt-dlp --skip-download --write-auto-subs --sub-langs en --sub-format vtt -o temp`
3. Parses the VTT file using a simple parser (port claude-video's `parse_vtt()`)
4. Returns `Transcript` with segments, or `None` if no captions found

The `observe` command flow becomes: try_captions() -> if None, fall back to local Whisper.
The `transcribe` command gets a `--prefer-captions` flag.

**VTT Parser to port:**
- `parse_vtt(path)` - reads WebVTT, extracts timestamped segments
- `_dedupe()` - collapses rolling duplicate cues from YouTube auto-subs
- `format_transcript()` - formats as `[MM:SS] text` lines

#### 3E. yt-dlp Video Download -> NEW `src/ophanim/core/download.py`

**Source:** claude-video `download.py`

**Why:** Currently ophanim only handles local files. claude-video's yt-dlp integration allows pasting a YouTube URL and having it downloaded automatically. This is the core UX improvement.

**How to integrate:**
Create `src/ophanim/core/download.py` with:
```python
def is_url(source: str) -> bool: ...
def download_video(url: str, out_dir: Path) -> dict: ...
def fetch_captions_only(url: str, out_dir: Path) -> dict: ...
```

Port from claude-video:
- `is_url()` - URL detection
- `download_url()` - yt-dlp download with format selection
- `fetch_captions()` - captions-only fetch (no video download)
- `_pick_subtitle()` - prefer English captions
- `_read_info()` - parse yt-dlp info.json

Add `yt-dlp` to dependencies in pyproject.toml.

**CLI integration:**
Update `observe` and `transcribe` commands to accept URLs:
```python
path: str = typer.Argument(..., help="Path or URL to video")
```

At the start of the command:
```python
if is_url(path):
    dl = download_video(path, work_dir)
    video_path = dl["video_path"]
    captions = dl.get("subtitle_path")
else:
    video_path = path
    captions = None
```

### What NOT to take

- **Groq/OpenAI Whisper API clients** - ophanim uses local faster-whisper (better privacy, no API key)
- **Claude Code skill infrastructure** - irrelevant
- **Build scripts** - irrelevant
- **The CLI interface** - different paradigm (typer vs argparse)

---

## Execution Order

1. Phase 1 first (restructure) -- everything else depends on clean layout
2. Phase 2 next (transcription fixes) -- independent of Phase 3
3. Phase 3 last (claude-video integration) -- depends on clean codebase from Phase 1

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Import breaks after restructure | All imports are `from ophanim.xxx` -- no changes needed. Test with `python -c "from ophanim.cli.app import app"` |
| GPU Whisper OOM with LM Studio | Auto-detect: only use GPU if >2GB free VRAM. Otherwise fall back to CPU. |
| yt-dlp not installed | Graceful fallback: if not available, skip URL download and captions. Print install instructions. |
| ffmpeg timeout still too low | 900s = 15 min. User said base model does 25-30 min in ~10 min. 50% headroom. |
| claude-video VTT parser incomplete | Port the parser + dedup. It handles YouTube auto-subs well (tested with 3.6k star repo). |

---

## File Change Summary

| File | Action | Phase |
|------|--------|-------|
| `pyproject.toml` | Update (src layout + yt-dlp dep) | 1 |
| `src/ophanim/` (entire dir) | Move from `ophanim/` | 1 |
| Root duplicates | DELETE | 1 |
| `src/ophanim/core/audio.py` | Timeout 300s -> 900s | 2 |
| `src/ophanim/providers/whisper.py` | GPU option + captions method | 2 |
| `src/ophanim/cli/commands/observe.py` | Add --prompt flag | 2 |
| `src/ophanim/cli/commands/transcribe.py` | Add --device flag | 2 |
| `src/ophanim/core/sampling.py` | Add ffmpeg dedup + auto-fps | 3 |
| `src/ophanim/core/video.py` | Add scene-change extraction | 3 |
| `src/ophanim/core/download.py` | NEW - yt-dlp integration | 3 |
| `src/ophanim/providers/whisper.py` | Add VTT parser + try_captions | 3 |
| `src/ophanim/cli/commands/observe.py` | URL support | 3 |
| `src/ophanim/cli/commands/transcribe.py` | URL support + --prefer-captions | 3 |
