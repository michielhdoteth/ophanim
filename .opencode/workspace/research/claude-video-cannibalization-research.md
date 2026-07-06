# Research: claude-video Cannibalization + Video Analysis Ecosystem
- Component: ophanim
- Created: 2026-07-05
- Status: complete

## Executive Summary

Deep-dived the claude-video repo (bradautomates/claude-video, 3.6k stars) file-by-file. Found **9 missed features** worth cannibalizing, ranked by value. Also surveyed the broader video analysis ecosystem (claude-real-video, yt-vision-pro, vidify, videolens, youtube-screenshot-extractor) and identified **6 additional external features** worth stealing. Total: 15 features ranked by value-to-effort ratio.

---

## Part 1: claude-video File-by-File Audit

### 1. frames.py — What We Missed

#### Feature: `auto_fps_focus()` — Dense Focused Extraction
- **What it does**: When a user specifies a time range (`--start`/`--end`), this function calculates a much denser frame budget. For a 5-second window, it targets 6 fps (30 frames). For 15 seconds, 4 fps (60 frames). For 30s, 60 frames. It recognizes that "zooming in" means the user wants detail, not a sparse scan.
- **ophanim status**: MISSING. ophanim's `auto_fps()` is one-size-fits-all — same budget whether you're scanning a full 30-min video or a 10-second clip someone wants to analyze closely. The `adaptive_sample()` in sampling.py does dense-then-sparse around scene changes, but has no concept of "user specified a focus window, go dense."
- **Worth adding**: HIGH. This is the single highest-value gap. When someone says "look at 2:15-2:45", ophanim currently gives them the same frame density as a full-video scan. Focus mode should be 4-6x denser.
- **Difficulty**: LOW. Pure math function. ~30 lines to add to `video.py`. Wire it into `observe.py` with `--start`/`--end` CLI flags.

#### Feature: `--start`/`--end` Focus Range CLI Flags
- **What it does**: Lets the user specify a time window to analyze. Frames outside the window are dropped. Transcript is filtered to the window. The report only covers the requested section.
- **ophanim status**: MISSING. `observe.py` has no `--start`/`--end` flags at all. It always processes the entire video.
- **Worth adding**: HIGH. Core UX feature. Without it, analyzing a specific moment in a long video requires pre-cutting the file.
- **Difficulty**: LOW. Add two typer options to `observe_cmd()`, pass to `extract_frames()` / `extract_scene_frames_ffmpeg()` with `-ss`/`-to` ffmpeg flags. Filter transcript segments. ~80 lines.

#### Feature: `--timestamps` Transcript-Cue Extraction
- **What it does**: Given comma-separated timestamps (e.g. `--timestamps 0:30,2:15,5:00`), extracts a single frame at each exact moment. These are "pinned" — never dropped by dedup or sampling. Designed for "look here" moments flagged in the transcript.
- **ophanim status**: MISSING. No way to extract frames at specific timestamps. `extract_frames_by_indices()` exists but takes frame indices, not human-readable timestamps.
- **Worth adding**: MEDIUM. Useful for targeted analysis ("show me what's on screen at 3:42"). Less critical than focus range.
- **Difficulty**: LOW. `extract_at_timestamps()` is ~60 lines of ffmpeg `-ss` per timestamp. Wire it as `--timestamps` flag.

#### Feature: Even-Sample `merge_frames()` with Pinned Frames
- **What it does**: Combines two frame lists (detail frames + pinned cue frames) into one chronological list, reindexes, and ensures pinned frames are never dropped by the cap.
- **ophanim status**: MISSING. ophanim has no concept of "pinned" vs "regular" frames.
- **Worth adding**: MEDIUM. Only matters if we add `--timestamps`.
- **Difficulty**: LOW. ~20 lines.

#### Feature: `parse_timestamps()` / `parse_time()` Robust Time Parser
- **What it does**: Parses `SS`, `MM:SS`, `HH:MM:SS` (with optional `.ms`) into float seconds. Used everywhere.
- **ophanim status**: PARTIAL. `_format_timestamp()` does MM:SS output. No robust input parser. `extract_audio_segment()` takes raw floats.
- **Worth adding**: MEDIUM. Quality-of-life for CLI. Enables `--start 1:30:00` syntax.
- **Difficulty**: TRIVIAL. ~25 lines, copy verbatim.

#### Feature: Frame Metadata `reason` Field
- **What it does**: Every frame dict includes a `reason` field: `"uniform"`, `"scene-change"`, `"keyframe"`, `"transcript-cue"`, `"first-frame"`. The markdown report shows why each frame was selected.
- **ophanim status**: MISSING. Frame dicts have `index`, `timestamp`, `image` but no provenance.
- **Worth adding**: LOW-MEDIUM. Debugging aid and report enrichment. Not critical.
- **Difficulty**: TRIVIAL. Add a string field to frame dicts.

#### Perceptual Dedup — Already Have It
- **What claude-video does**: `dedupe_perceptual()` uses ffmpeg to create 16x16 grayscale thumbnails, then computes mean absolute per-pixel difference. Compares against last *kept* frame (not previous frame) to catch slow fades.
- **ophanim status**: HAVE IT. `dedupe_frames_ffmpeg()` does the same 16x16 thumbnail approach via `np.mean(np.abs(...))`. The only difference: ophanim compares against the immediately previous frame, while claude-video compares against the last *kept* frame. The "last kept" approach is slightly better for catching slow fades.
- **Worth upgrading**: LOW. One-line change to track `prev_thumb` as the last kept, not last seen.
- **Difficulty**: TRIVIAL.

### 2. watch.py — What We Missed

#### Feature: Detail Mode Dial (`--detail transcript|efficient|balanced|token-burner`)
- **What it does**: Four preset modes that control the speed/fidelity tradeoff:
  - `transcript`: No frames at all, captions only. Zero video processing cost.
  - `keyframe` (efficient): `skip_frame nokey`, cap 50. ~0.5s extraction.
  - `scene-change` (balanced): Full scene detection, cap 100. ~20s extraction.
  - `scene-change` (token-burner): Full scene detection, uncapped. Keeps every cut.
- **ophanim status**: PARTIAL. `observe.py` has `--mode fast|balanced|detailed` but these control FPS/resolution/max_frames, not the extraction *engine*. ophanim always does the same extraction regardless of mode.
- **Worth adding**: HIGH. The "transcript" mode is brilliant — for captioned videos, you can skip frame extraction entirely and save all processing time. The "efficient" keyframe mode is 40x faster than scene detection.
- **Difficulty**: MEDIUM. ~100 lines to wire up the mode-to-engine mapping in `observe.py`. The engines already exist in `video.py` (`extract_keyframes_ffmpeg`, `extract_scene_frames_ffmpeg`).

#### Feature: Markdown Report Generation
- **What it does**: `watch.py` prints a structured markdown report to stdout with: source URL, title, uploader, duration, focus range, resolution, detail mode, frame count/candidates/dedup stats, frame paths with timestamps and reasons, transcript source and text. Designed to be consumed by Claude's Read tool.
- **ophanim status**: PARTIAL. `observe.py` saves `observations.json`, `summary.md`, `timeline.md`, and `transcript.txt` as separate files. Rich console output exists but no unified markdown report.
- **Worth adding**: MEDIUM. A single markdown report is more useful than scattered files when piping to an LLM. But ophanim's JSON output is more machine-readable.
- **Difficulty**: LOW. ~80 lines to generate a report string.

#### Feature: Sparse Scan Warning for Long Videos
- **What it does**: When a video >10 min is processed with capped modes, the report prints: "This is a N-minute video. Frame coverage is sparse at this length under `balanced` detail — re-run focused or use `token-burner`."
- **ophanim status**: MISSING. `observe.py` prints a cost warning but no actionable guidance.
- **Worth adding**: LOW. Nice-to-have UX. 5 lines.
- **Difficulty**: TRIVIAL.

#### Feature: Caption-First Optimization (Skip Download When Possible)
- **What it does**: For transcript-only mode, `watch.py` checks captions first via `fetch_captions()` (metadata + VTT only, no video download). If captions exist and mode is `transcript`, it never downloads the video file at all.
- **ophanim status**: MISSING. `observe.py` always processes the full video file. No caption-first path.
- **Worth adding**: MEDIUM. Saves bandwidth and time for captioned YouTube videos when you only need the transcript.
- **Difficulty**: MEDIUM. Requires yt-dlp integration (see download.py section).

### 3. download.py — URL/Download Support

#### Feature: yt-dlp Integration for URL Downloads
- **What it does**: Full yt-dlp wrapper supporting:
  - URL detection (`is_url()`)
  - Caption fetching without video download (`fetch_captions()`)
  - Audio-only downloads for Whisper (`audio_only=True`)
  - Quality selection (`bv*[height<=720]+ba/b`)
  - Info JSON metadata extraction
  - Subtitle language selection and VTT conversion
  - Local file passthrough
- **ophanim status**: **COMPLETELY MISSING**. ophanim only works with local files. No URL support, no yt-dlp, no caption fetching.
- **Worth adding**: HIGH. This is the biggest architectural gap. ophanim is a local-only tool; claude-video works with any URL. YouTube, TikTok, Loom, Vimeo, etc.
- **Difficulty**: HIGH. Requires adding yt-dlp as a dependency, building the download pipeline, handling captions, error recovery. ~200-300 lines. But it's mostly wrapping yt-dlp CLI calls.

### 4. transcribe.py — VTT Parsing

#### Feature: VTT Parsing with YouTube Auto-Sub Dedup
- **What it does**: Parses WebVTT subtitle files into `{start, end, text}` segments. Includes `_dedupe()` that collapses rolling-duplicate cues common in YouTube auto-subs (each line appears 2-3 times as it scrolls). Also has `filter_range()` for focus windows and `format_transcript()` for human-readable output.
- **ophanim status**: MISSING. ophanim has `WhisperProvider` for local ASR but no VTT/caption parsing. If a video has embedded captions, ophanim ignores them and re-transcribes from scratch.
- **Worth adding**: HIGH. YouTube auto-captions are free, instant, and good enough for most use cases. Re-transcribing with Whisper is slow and costs VRAM.
- **Difficulty**: LOW. `parse_vtt()` is ~50 lines of stdlib regex. `filter_range()` is ~5 lines. Copy verbatim.

#### Feature: Whisper API (Groq/OpenAI) as Fallback
- **What it does**: `whisper.py` extracts mono 16kHz 64kbps mp3 (~480 kB/min), uploads to Groq (whisper-large-v3, preferred) or OpenAI (whisper-1) via raw HTTP multipart. Includes chunked upload for files >24MB, retry logic with exponential backoff, rate limit handling. Pure stdlib — no pip packages.
- **ophanim status**: HAVE LOCAL WHISPER. ophanim uses `faster-whisper` locally with GPU auto-detection. This is actually *better* than cloud APIs for privacy and cost. But ophanim has no fallback when local Whisper fails or VRAM is low.
- **Worth adding**: LOW-MEDIUM. Could add Groq/OpenAI as a fallback when local Whisper is unavailable. But this adds API key management complexity. The local approach is philosophically aligned with ophanim's local-first design.
- **Difficulty**: MEDIUM. ~150 lines for the HTTP client, but API key management is annoying.

### 5. Other Files

#### Feature: `config.py` — Persistent Configuration
- **What it does**: Reads `~/.config/watch/.env` for default settings (detail mode, API keys). Supports inline comments, quoted values, env var overrides.
- **ophanim status**: HAVE IT. `storage/config.py` handles ophanim's config. Different format but same concept.
- **Worth adding**: NO. Already covered.

#### Feature: `setup.py` — Preflight Dependency Check
- **What it does**: Checks for ffmpeg, yt-dlp, Whisper API keys on first run. Auto-installs via Homebrew on macOS. Scaffolds config file with commented placeholders.
- **ophanim status**: MISSING. No dependency checker or auto-installer.
- **Worth adding**: LOW. Nice for onboarding but not critical for the library itself.
- **Difficulty**: LOW. ~80 lines.

---

## Part 2: External Repos — Additional Features Worth Cannibalizing

### From claude-real-video (HUANGCHIHHUNGLeo, 1k stars)

#### Feature: Sliding-Window Dedup (Dedup Window > 1)
- **What it does**: `--dedup-window 4` compares each frame against the last 4 kept frames, not just the last 1. This catches A-B-A cutaways — if frame A appears, then B, then A again, the second A is dropped because it matches a frame in the window.
- **ophanim status**: MISSING. Both ophanim and claude-video's default mode only compare against the last kept frame.
- **Worth adding**: MEDIUM. Prevents redundant frames in videos with repeated shots (tutorials that cut back to the same slide, news segments that reuse B-roll).
- **Difficulty**: LOW. Change `_make_thumbnail` comparison to loop over last N kept thumbnails. ~15 lines.

#### Feature: `--report` Keep/Decision Visualization
- **What it does**: `--report` saves dropped frames to `./dropped/` and writes `report.html` showing every keep/drop decision with diff percentages. Helps tune dedup threshold.
- **ophanim status**: MISSING. No visual debugging for dedup decisions.
- **Worth adding**: LOW. Developer tuning aid, not end-user feature.
- **Difficulty**: MEDIUM. ~100 lines for HTML generation.

#### Feature: `--why` Analysis Lens
- **What it does**: `--why "find the pricing strategy"` writes the goal into MANIFEST.txt so the LLM analyzes with that specific lens instead of a generic summary.
- **ophanim status**: MISSING. ophanim has `--question` which is similar but less integrated into the output manifest.
- **Worth adding**: LOW. ophanim's `--question` already does this conceptually.
- **Difficulty**: TRIVIAL. Pass string through to output.

#### Feature: `--kb` Knowledge Base Integration
- **What it does**: `--kb ~/notes` saves the analysis as a dated markdown note into a specified folder (e.g. Obsidian vault), so it joins the user's knowledge base.
- **ophanim status**: MISSING. `_save_memory_md()` exists but saves to a fixed `memory/videos/` directory.
- **Worth adding**: LOW. Quality-of-life for note-takers.
- **Difficulty**: TRIVIAL. Accept a path argument, write there instead of default.

#### Feature: `--cookies` Authenticated Downloads
- **What it does**: Passes a Netscape cookie file to yt-dlp for login-gated content (private Vimeo, age-restricted YouTube, etc.).
- **ophanim status**: N/A (no yt-dlp support at all).
- **Worth adding**: MEDIUM (once yt-dlp is added).
- **Difficulty**: TRIVIAL. One flag to pass through to yt-dlp.

### From yt-vision-pro (ktnCodes)

#### Feature: OCR-First Frame Filtering
- **What it does**: Runs RapidOCR on every extracted frame *before* deduplication. Frames with OCR text are flagged as "slides" and get looser dedup thresholds (a slide with one bullet point added is kept, not collapsed).
- **ophanim status**: MISSING. ophanim's dedup is purely visual — a slide gaining a bullet point might survive (pixel difference) or might not, depending on the threshold.
- **Worth adding**: MEDIUM. For presentation/tutorial analysis, OCR-aware dedup preserves educational content that visual-only dedup would collapse.
- **Difficulty**: HIGH. Requires adding RapidOCR dependency, running OCR on every frame, modifying dedup logic. ~150 lines + new dependency.

#### Feature: Chapter-Aware Processing
- **What it does**: Parses YouTube chapter markers (or generates synthetic 15-min chunks). Processes each chapter independently with its own manifest. Prevents context overflow on 2+ hour videos.
- **ophanim status**: MISSING. ophanim processes the entire video as one unit.
- **Worth adding**: MEDIUM. For long-form content (lectures, podcasts), chapter-aware processing produces better summaries and prevents token overflow.
- **Difficulty**: HIGH. Requires parsing chapter metadata from yt-dlp, splitting processing, merging results. ~200 lines.

#### Feature: Resumable Pipeline Stages
- **What it does**: Each pipeline stage (fetch, extract, ocr, dedup, align, manifest) writes a sentinel file. On re-run, completed stages are skipped. `--force` clears all sentinels. `--from-stage` re-runs from a specific point.
- **ophanim status**: PARTIAL. `RunCache` has caching but not at the stage level.
- **Worth adding**: MEDIUM. For long videos that take minutes to process, resumability saves time on failures.
- **Difficulty**: MEDIUM. ~80 lines to add sentinel-based stage tracking.

### From youtube-screenshot-extractor (EnragedAntelope, 54 stars)

#### Feature: Quality/Blur Filtering
- **What it does**: Filters frames by quality score (0-100) and blur threshold. Also detects and optionally removes black bars. Post-processing filters (gradfun, deblock, deband) reduce compression artifacts.
- **ophanim status**: MISSING. ophanim extracts whatever ffmpeg gives it, including blurry/dark/letterboxed frames.
- **Worth adding**: LOW. Useful for dataset preparation but not for video understanding.
- **Difficulty**: MEDIUM. Requires Laplacian variance for blur detection, ~60 lines.

#### Feature: GPU-Accelerated Extraction
- **What it does**: `--use-gpu` uses ffmpeg's CUDA-accelerated decoding and scaling.
- **ophanim status**: PARTIAL. ophanim has GPU detection for Whisper but not for ffmpeg frame extraction.
- **Worth adding**: LOW. Marginal speedup for frame extraction which is already fast.
- **Difficulty**: LOW. Add `-hwaccel cuda` to ffmpeg commands when GPU available. ~10 lines.

### From vidify (shepnerd)

#### Feature: FAISS Index for Video Q&A
- **What it does**: Builds a FAISS index over transcript, frames, and metadata. Enables evidence-backed Q&A — "what did she say about pricing?" retrieves the relevant frames + transcript segments.
- **ophanim status**: MISSING. ophanim's `ask` command exists but does simple prompt-based Q&A, not retrieval-augmented.
- **Worth adding**: LOW-MEDIUM. Useful for very long videos where the VLM context window can't hold all frames. But adds FAISS dependency.
- **Difficulty**: HIGH. Requires FAISS, embedding model, index construction, retrieval pipeline. ~300 lines.

#### Feature: Clip Export / Highlight Detection
- **What it does**: Detects highlight moments and optionally exports video clips. Assembles reels from highlights.
- **ophanim status**: MISSING. ophanim is read-only — it analyzes but doesn't edit/cut video.
- **Worth adding**: LOW. Out of scope for a video understanding tool.
- **Difficulty**: HIGH. ffmpeg complex filter graphs.

---

## Part 3: Ranked Feature List

### Tier 1 — High Value, Low Effort (Do First)

| # | Feature | Source | Value | Effort | Lines |
|---|---------|--------|-------|--------|-------|
| 1 | `auto_fps_focus()` dense extraction for focus ranges | claude-video frames.py | HIGH | LOW | ~30 |
| 2 | `--start`/`--end` focus range CLI flags | claude-video watch.py | HIGH | LOW | ~80 |
| 3 | VTT parsing with YouTube auto-sub dedup | claude-video transcribe.py | HIGH | LOW | ~50 |
| 4 | `parse_time()` robust timestamp parser | claude-video frames.py | MEDIUM | TRIVIAL | ~25 |
| 5 | Frame `reason` field for provenance | claude-video frames.py | LOW-MED | TRIVIAL | ~10 |
| 6 | Dedup against last *kept* frame (not last seen) | claude-video frames.py | LOW | TRIVIAL | ~5 |

**Total: ~200 lines for 6 features. All pure additions, no refactoring.**

### Tier 2 — High Value, Medium Effort (Do Next)

| # | Feature | Source | Value | Effort | Lines |
|---|---------|--------|-------|--------|-------|
| 7 | Detail mode dial (`--detail transcript\|efficient\|balanced`) | claude-video watch.py | HIGH | MEDIUM | ~100 |
| 8 | `--timestamps` cue extraction | claude-video frames.py | MEDIUM | LOW | ~60 |
| 9 | yt-dlp URL download support | claude-video download.py | HIGH | HIGH | ~250 |
| 10 | Sliding-window dedup (window > 1) | claude-real-video | MEDIUM | LOW | ~15 |
| 11 | Sparse scan warning for long videos | claude-video watch.py | LOW | TRIVIAL | ~5 |

**Total: ~430 lines for 5 features. #9 is the big one.**

### Tier 3 — Medium Value, High Effort (Future)

| # | Feature | Source | Value | Effort | Lines |
|---|---------|--------|-------|--------|-------|
| 12 | Chapter-aware processing | yt-vision-pro | MEDIUM | HIGH | ~200 |
| 13 | Resumable pipeline stages | yt-vision-pro | MEDIUM | MEDIUM | ~80 |
| 14 | OCR-first frame filtering | yt-vision-pro | MEDIUM | HIGH | ~150 |
| 15 | FAISS index for retrieval Q&A | vidify | LOW-MED | HIGH | ~300 |

**Total: ~730 lines for 4 features. These are architectural changes.**

---

## Part 4: What ophanim Already Has (That claude-video Doesn't)

For completeness, features ophanim has that claude-video lacks:

| Feature | ophanim | claude-video |
|---------|---------|--------------|
| Local Whisper (faster-whisper, GPU) | YES | NO (cloud only) |
| OpenCV-based scene detection (HSV histogram) | YES | NO (ffmpeg only) |
| GPU auto-downgrade for VRAM management | YES | NO |
| Structured JSON output (Pydantic models) | YES | NO (markdown only) |
| Caching system (RunCache) | YES | NO (re-processes every time) |
| Multi-provider VLM support (LM Studio, SAM) | YES | NO (Claude only) |
| Image observation (single frame) | YES | NO (video only) |
| Entity extraction from timeline | YES | NO |
| Memory/notes saving | YES | YES (--kb) |
| Adaptive scene-aware sampling | YES | YES |

---

## Recommendation

**Phase 4A (Quick Wins — 200 lines):** Implement Tier 1 features. Focus range, VTT parsing, and dense focused extraction are the highest-value additions that require minimal code. These alone make ophanim competitive with claude-video for local analysis.

**Phase 4B (URL Support — 250 lines):** Add yt-dlp integration. This is the single biggest architectural gap. Without it, ophanim can only analyze files you already have. With it, you can paste any YouTube/TikTok/Vimeo URL and get analysis.

**Phase 4C (Polish — 180 lines):** Detail modes, timestamp cues, sliding-window dedup. These make the tool more configurable and robust.

**Skip for now:** OCR filtering, chapter-aware processing, FAISS index, clip export. These are either out of scope or require significant new dependencies.

## Sources

1. https://github.com/bradautomates/claude-video (main repo, 3.6k stars)
2. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/frames.py (source)
3. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/watch.py (source)
4. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/download.py (source)
5. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/transcribe.py (source)
6. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/whisper.py (source)
7. https://github.com/bradautomates/claude-video/blob/main/skills/watch/scripts/config.py (source)
8. https://github.com/HUANGCHIHHUNGLeo/claude-real-video (1k stars, sliding-window dedup, --report, --why, --kb)
9. https://github.com/ktnCodes/yt-vision-pro (OCR-first filtering, chapter-aware, resumable pipeline)
10. https://github.com/shepnerd/vidify (FAISS index, clip export, live stream)
11. https://github.com/EnragedAntelope/youtube-screenshot-extractor (54 stars, quality/blur filtering, GPU accel)
12. https://github.com/shadoprizm/videolens (bug/meeting analysis modes, timeline evidence)
13. https://github.com/docusphere/video-analyzer (.avt format, Gemini-based)
14. ophanim source: src/ophanim/core/video.py, src/ophanim/core/sampling.py, src/ophanim/cli/commands/observe.py
