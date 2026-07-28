# Plan: Transform Ophanim into Open Vision (ACLI Tool)

- Component: openvision (root)
- Task ID: openvision-transformation
- Created: 2026-07-27
- Status: pending

## Overview

Transform "ophanim" (v0.1.1) — a personal local vision CLI — into "Open Vision" (v1.0.0) — a self-hosted, privacy-first ACLI (AI Command Line Interface) that any AI agent or human can use with vision models. The tool must support multiple VLM backends (LM Studio, Ollama, llama.cpp, any OpenAI-compatible API), work without cloud dependencies, and provide structured outputs suitable for programmatic consumption.

## Current Architecture Summary

```
ophanim/
  __init__.py          # Version "0.1.1"
  __main__.py          # Entry point (python -m ophanim)
  models.py            # Pydantic schemas (ObserveResult, AskResult, etc.)
  config/default.yaml  # Default config (LM Studio only)
  cli/
    app.py             # Typer app, name="ophanim"
    commands/          # 8 commands: probe, observe, ask, segment, track, status, memory, transcribe
  core/
    video.py, image.py, sampling.py, gpu.py, errors.py, audio.py, captions.py, download.py
  providers/
    base.py            # VlmProvider ABC (describe_image, describe_frames, check_health)
    lmstudio.py        # LmStudioProvider (OpenAI-compatible)
    sam.py             # SamProvider, Sam3Provider (segmentation)
    whisper.py         # WhisperProvider (transcription)
    diarizer.py        # Speaker diarization
  storage/
    config.py, cache.py, paths.py  # Config, cache, stable paths (~/.ophanim)
```

**Key observations:**
- `VlmProvider` base class already exists with clean ABC interface — extension point is ready
- All providers use `from ophanim.xxx` imports — package rename touches ~100 import statements
- `LmStudioProvider` already uses OpenAI-compatible API — generic provider can inherit most logic
- CLI commands hardcode `LmStudioProvider` — need factory/dispatch pattern
- Config uses `OPHANIM_HOME`, `OPHANIM_CONFIG` env vars — need migration path
- `~/.ophanim` is the data home — need backward compat or migration

---

## Phase 1: Rename and Rebrand (Foundation)

### Task 1.1: Rename Package Directory
- **Description:** Rename `ophanim/` directory to `openvision/`
- **Acceptance criteria:**
  - `openvision/` directory exists with all subpackages
  - `ophanim/` directory removed
  - All `from ophanim.xxx` imports become `from openvision.xxx`
- **Dependencies:** None
- **Estimate:** Small (mechanical)
- **Risk:** LOW — straightforward find-replace

### Task 1.2: Update All Python Imports
- **Description:** Replace every `ophanim` import reference across all .py files
- **Files affected:** ~30+ Python files with `from ophanim.xxx` or `import ophanim`
- **Acceptance criteria:**
  - `grep -r "from ophanim" openvision/` returns zero matches
  - `grep -r "import ophanim" openvision/` returns zero matches
  - All existing tests pass after rename
- **Dependencies:** Task 1.1
- **Estimate:** Small
- **Risk:** LOW

### Task 1.3: Update pyproject.toml
- **Description:** Rename package, entry point, description, version bump to 1.0.0
- **Changes:**
  ```toml
  name = "openvision"
  version = "1.0.0"
  description = "Self-hosted AI vision CLI — let any AI see any video, image, or visual content"
  
  [project.scripts]
  openvision = "openvision.__main__:main"
  ov = "openvision.__main__:main"  # Short alias
  
  [tool.setuptools.packages.find]
  include = ["openvision*"]
  ```
- **Acceptance criteria:**
  - `pip install -e .` installs as `openvision` command
  - `openvision --version` works
  - `ov --version` works (short alias)
- **Dependencies:** Task 1.1
- **Estimate:** Small
- **Risk:** LOW

### Task 1.4: Update Error Classes
- **Description:** Rename `OphanimError` to `OpenVisionError` throughout
- **Files:** `core/errors.py`, `cli/app.py`
- **Acceptance criteria:**
  - All error handling uses `OpenVisionError`
  - Backward compat alias: `OphanimError = OpenVisionError` with deprecation warning
- **Dependencies:** Task 1.1
- **Estimate:** Small
- **Risk:** LOW

### Task 1.5: Update Config and Path References
- **Description:** Update env vars, default paths, config references
- **Changes:**
  - `OPHANIM_HOME` → `OPENVISION_HOME` (with `OPHANIM_HOME` fallback)
  - `OPHANIM_CONFIG` → `OPENVISION_CONFIG` (with `OPHANIM_CONFIG` fallback)
  - `~/.ophanim` → `~/.openvision` (with `~/.ophanim` migration)
  - `DEFAULT_HOME_NAME = ".openvision"`
  - Config file comments updated
- **Acceptance criteria:**
  - Fresh install uses `~/.openvision`
  - Existing `~/.ophanim` data is auto-migrated or symlinked
  - Both env vars work (new preferred, old deprecated)
- **Dependencies:** Task 1.1
- **Estimate:** Small
- **Risk:** MEDIUM — migration logic needed

### Task 1.6: Update README and Documentation
- **Description:** Rewrite README.md for Open Vision branding
- **Changes:**
  - New name, description, command examples
  - Update all `ophanim` references to `openvision`
  - Update architecture diagram
  - Update data paths table
  - Update Quick Start section
  - Add multi-provider setup instructions
- **Acceptance criteria:**
  - No remaining "ophanim" references in README
  - All command examples work with new name
  - New features documented
- **Dependencies:** Tasks 1.1-1.5
- **Estimate:** Medium
- **Risk:** LOW

### Task 1.7: Update .gitignore
- **Description:** Update ignore patterns for new naming
- **Changes:**
  - `ophanim_test_output.json` → `openvision_test_output.json`
  - `ophanim/config/local.yaml` → `openvision/config/local.yaml`
  - Comment updates
- **Acceptance criteria:** Patterns match new directory structure
- **Dependencies:** Task 1.1
- **Estimate:** Small
- **Risk:** LOW

### Task 1.8: Rename GitHub Repository
- **Description:** Use `gh` CLI to rename the repository
- **Command:**
  ```bash
  gh repo rename open-vision --repo <owner>/ophanim
  ```
- **Acceptance criteria:**
  - GitHub repo accessible at new URL
  - All remote URLs updated
- **Dependencies:** All Phase 1 tasks committed
- **Estimate:** Small
- **Risk:** MEDIUM — breaks existing clones/URLs

---

## Phase 2: Multi-Provider Architecture

### Task 2.1: Create Provider Registry and Factory
- **Description:** Build a provider registry that auto-detects and instantiates VLM providers
- **New file:** `openvision/providers/registry.py`
- **Design:**
  ```python
  class ProviderRegistry:
      """Auto-discover and instantiate VLM providers."""
      
      PROVIDERS = {
          "lmstudio": LmStudioProvider,
          "ollama": OllamaProvider,
          "llamacpp": LlamaCppProvider,
          "openai": OpenAIProvider,
          "auto": None,  # Auto-detect
      }
      
      @classmethod
      def create(cls, config: dict) -> VlmProvider:
          """Create provider from config, with auto-detection."""
          
      @classmethod
      def detect(cls) -> dict:
          """Probe common endpoints, return available providers."""
  ```
- **Acceptance criteria:**
  - `ProviderRegistry.create({"provider": "ollama", ...})` returns OllamaProvider
  - `ProviderRegistry.detect()` probes localhost:1234, localhost:11434, localhost:8080
  - Auto-detection finds first available provider
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

### Task 2.2: Implement OllamaProvider
- **Description:** Native Ollama integration using `/api/chat` endpoint
- **New file:** `openvision/providers/ollama.py`
- **Design:**
  - Base URL: `http://localhost:11434`
  - Uses `/api/chat` with `images` field (base64)
  - Supports `keep_alive` for model persistence
  - Health check: `GET /api/tags`
  - Auto-pull model if not available
- **Acceptance criteria:**
  - `OllamaProvider({"model": "llava:13b"})` works
  - `describe_image()` sends base64 via Ollama API
  - `check_health()` returns True when Ollama running
  - Error handling for connection failures
- **Dependencies:** Task 2.1
- **Estimate:** Medium
- **Risk:** LOW — well-documented API

### Task 2.3: Implement LlamaCppProvider
- **Description:** llama.cpp server integration via OpenAI-compatible endpoint
- **New file:** `openvision/providers/llamacpp.py`
- **Design:**
  - Base URL: `http://localhost:8080/v1`
  - Uses OpenAI-compatible `/v1/chat/completions`
  - llama.cpp specific: handles `clip_model` loading
  - Health check: `GET /v1/models`
- **Acceptance criteria:**
  - `LlamaCppProvider({"model": "llava-v1.5-7b", "base_url": "http://localhost:8080/v1"})` works
  - Inherits from generic OpenAI-compatible base
  - Health check works
- **Dependencies:** Task 2.4
- **Estimate:** Small
- **Risk:** LOW — llama.cpp serves OpenAI-compatible API

### Task 2.4: Implement Generic OpenAIProvider
- **Description:** Generic provider for any OpenAI-compatible API (OpenAI, Together, Groq, etc.)
- **New file:** `openvision/providers/openai_generic.py`
- **Design:**
  - Extends `LmStudioProvider` with API key support
  - Configurable base_url for any endpoint
  - API key via config or `OPENAI_API_KEY` env var
  - Model name passed as-is
  - Supports both vision and text-only endpoints
- **Acceptance criteria:**
  - Works with: OpenAI, Together AI, Groq, vLLM, LocalAI, AnyScale
  - API key auth works
  - No hardcoded model names
  - Proper error messages for auth failures
- **Dependencies:** None (extends existing LmStudioProvider)
- **Estimate:** Medium
- **Risk:** LOW

### Task 2.5: Refactor LmStudioProvider to Share with OpenAIProvider
- **Description:** Extract common OpenAI-compatible logic into a shared base
- **Refactoring:**
  ```
  LmStudioProvider (no auth, localhost:1234)
      └── inherits from OpenAICompatibleProvider
  
  OpenAIProvider (API key, configurable URL)
      └── inherits from OpenAICompatibleProvider
  
  LlamaCppProvider (no auth, localhost:8080)
      └── inherits from OpenAICompatibleProvider
  ```
- **Acceptance criteria:**
  - DRY: shared `_send_chat_completion()`, `_parse_usage()` methods
  - Each subclass only overrides config/defaults
  - All existing LM Studio tests pass
- **Dependencies:** Tasks 2.3, 2.4
- **Estimate:** Medium
- **Risk:** MEDIUM — must not break existing LM Studio behavior

### Task 2.6: Update Config Schema for Multi-Provider
- **Description:** Extend `config/default.yaml` to support all providers
- **New config structure:**
  ```yaml
  models:
    vlm:
      provider: "auto"  # auto | lmstudio | ollama | llamacpp | openai
      model: "auto"     # auto-detect or explicit name
      
      # Provider-specific settings (optional, with sensible defaults)
      lmstudio:
        base_url: "http://localhost:1234/v1"
      
      ollama:
        base_url: "http://localhost:11434"
        keep_alive: "5m"
      
      llamacpp:
        base_url: "http://localhost:8080/v1"
      
      openai:
        base_url: null  # Use official API
        api_key: null   # Or OPENAI_API_KEY env var
  ```
- **Acceptance criteria:**
  - Backward compatible: existing configs still work
  - New providers have sensible defaults
  - `provider: "auto"` triggers auto-detection
- **Dependencies:** Tasks 2.1-2.5
- **Estimate:** Medium
- **Risk:** LOW

### Task 2.7: Refactor CLI Commands to Use Provider Factory
- **Description:** Replace hardcoded `LmStudioProvider` with `ProviderRegistry.create()`
- **Files:** All 8 CLI commands (observe, ask, segment, track, probe, status, memory, transcribe)
- **Pattern:**
  ```python
  # Before (hardcoded)
  provider = LmStudioProvider(vlm_config)
  
  # After (factory)
  from openvision.providers.registry import ProviderRegistry
  provider = ProviderRegistry.create(config.get("models", {}).get("vlm", {}))
  ```
- **Acceptance criteria:**
  - All commands work with any configured provider
  - `--provider` flag available on commands that use VLM
  - `openvision observe video.mp4 --provider ollama --model llava:13b` works
- **Dependencies:** Tasks 2.1, 2.6
- **Estimate:** Medium
- **Risk:** MEDIUM — touches all command files

### Task 2.8: Add Provider Health Check to Status Command
- **Description:** Show all configured providers and their status in `openvision status`
- **New output:**
  ```
  Providers:
    ✓ LM Studio   (http://localhost:1234/v1) — google/gemma-4-e2b
    ✓ Ollama      (http://localhost:11434) — llava:13b
    ✗ llama.cpp    (http://localhost:8080/v1) — not running
    — OpenAI      (not configured)
  ```
- **Acceptance criteria:**
  - `openvision status` shows provider health
  - `openvision status --json` includes provider info
  - Health checks use short timeouts (5s)
- **Dependencies:** Tasks 2.1, 2.7
- **Estimate:** Small
- **Risk:** LOW

---

## Phase 3: Enhanced Vision Capabilities

### Task 3.1: Add Image URL Fetching
- **Description:** Support analyzing images from URLs (HTTP/HTTPS)
- **New file:** `openvision/core/web.py`
- **Design:**
  - Fetch image from URL, decode with OpenCV/PIL
  - Support common formats: JPEG, PNG, WebP, GIF
  - Timeout and error handling
  - Cache fetched images temporarily
- **Usage:**
  ```bash
  openvision observe "https://example.com/photo.jpg"
  openvision ask "https://example.com/chart.png" "What are the numbers?"
  ```
- **Acceptance criteria:**
  - URL inputs auto-detected (like yt-dlp URLs)
  - Images fetched, decoded, and processed
  - Timeout after 30s with clear error
  - Works with `observe` and `ask` commands
- **Dependencies:** None
- **Estimate:** Small
- **Risk:** LOW

### Task 3.2: Add Image Folder Batch Processing
- **Description:** Process all images in a directory
- **New command:** `openvision batch <directory> [question]`
- **Design:**
  - Scan directory for image files (jpg, png, webp, etc.)
  - Process each through VLM
  - Aggregate results into single JSON output
  - Optional parallel processing (configurable workers)
- **Acceptance criteria:**
  - `openvision batch ./photos/ --json` outputs array of results
  - Progress bar shows processing status
  - Skips non-image files gracefully
  - Supports `--max-workers N` for parallelism
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

### Task 3.3: Add Webcam/Screen Capture Analysis
- **Description:** Capture from webcam or screen and analyze in real-time
- **New commands:**
  ```bash
  openvision capture webcam    # Single webcam frame
  openvision capture screen    # Single screenshot
  openvision capture screen --continuous --interval 5  # Every 5 seconds
  ```
- **Design:**
  - Webcam: OpenCV `VideoCapture(0)`
  - Screen: `mss` library (cross-platform) or `PIL.ImageGrab`
  - `--continuous` mode for monitoring
  - Output as JSON for programmatic use
- **Acceptance criteria:**
  - Single capture works on Windows/Mac/Linux
  - Continuous mode respects interval
  - Graceful handling when no webcam available
  - Screen capture works on all platforms
- **Dependencies:** None
- **Estimate:** Large
- **Risk:** MEDIUM — platform-specific screen capture

### Task 3.4: Improve Segment Command with More SAM Variants
- **Description:** Extend SAM provider with additional model options
- **Changes:**
  - Add `sam-2.1-hiera-large` model option
  - Add `sam-2.1-hiera-tiny` for low-VRAM
  - Add `grounded-sam` for text+box prompts
  - Auto-select based on VRAM
- **Acceptance criteria:**
  - `openvision segment video.mp4 "person" --model sam-2.1-hiera-large` works
  - Auto-selection picks appropriate model based on VRAM
  - All existing SAM tests pass
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

### Task 3.5: Add Structured JSON Output to All Commands
- **Description:** Ensure every command produces clean, parseable JSON with `--json`
- **Standard JSON envelope:**
  ```json
  {
    "status": "success",
    "command": "observe",
    "version": "1.0.0",
    "data": { ... },
    "tokens": { "prompt": 0, "completion": 0, "total": 0 },
    "timing": { "elapsed_ms": 1234 }
  }
  ```
- **Acceptance criteria:**
  - All commands support `--json`
  - JSON is valid and parseable
  - Error responses also have structured JSON
  - Schema documented in README
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

---

## Phase 4: Self-Hosted Setup

### Task 4.1: Create Setup Wizard
- **Description:** Interactive first-time setup command
- **New command:** `openvision setup`
- **Flow:**
  1. Detect available backends (LM Studio, Ollama, llama.cpp)
  2. Guide user through configuration
  3. Download recommended vision model (if desired)
  4. Test connection and model loading
  5. Save config to `~/.openvision/config.yaml`
- **Acceptance criteria:**
  - Detects running services automatically
  - Offers to download models (with size warnings)
  - Tests with a sample image
  - Saves working configuration
  - Skippable with `--non-interactive` for CI
- **Dependencies:** Phase 2 complete
- **Estimate:** Large
- **Risk:** MEDIUM — model download can be slow/large

### Task 4.2: Add Automatic Model Downloading
- **Description:** Download and cache vision models automatically
- **New file:** `openvision/core/model_manager.py`
- **Design:**
  - Detect Ollama → `ollama pull` integration
  - Detect LM Studio → show download instructions
  - GGUF model download support (for llama.cpp)
  - Progress tracking for large downloads
  - Model cache under `~/.openvision/models/`
- **Acceptance criteria:**
  - `openvision setup --model llava:7b` downloads via Ollama
  - Download progress shown
  - Skip if model already cached
  - Handle disk space checks
- **Dependencies:** Task 4.1
- **Estimate:** Large
- **Risk:** MEDIUM — download failures, disk space

### Task 4.3: Create Docker Container
- **Description:** Dockerfile for easy deployment
- **New files:** `Dockerfile`, `docker-compose.yml`
- **Design:**
  ```dockerfile
  FROM python:3.12-slim
  # Install ffmpeg, OpenCV dependencies
  COPY . /app
  WORKDIR /app
  RUN pip install -e .
  EXPOSE 8000  # Future: API mode
  ENTRYPOINT ["openvision"]
  ```
- **docker-compose.yml:**
  - Ollama service (with GPU passthrough)
  - Open Vision service
  - Volume mounts for models and data
- **Acceptance criteria:**
  - `docker-compose up` starts both Ollama and Open Vision
  - GPU passthrough works (nvidia-docker)
  - Persistent data via volumes
  - Health checks included
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

### Task 4.4: Add Health Checks and Diagnostics
- **Description:** Comprehensive system diagnostics
- **New command:** `openvision doctor`
- **Checks:**
  - Python version (>=3.12)
  - ffmpeg installed and in PATH
  - GPU detected and VRAM sufficient
  - At least one VLM provider available
  - Model loaded/available
  - Disk space sufficient
  - Network connectivity (for downloads)
- **Acceptance criteria:**
  - `openvision doctor` shows all checks
  - Pass/fail with remediation suggestions
  - `--json` output for programmatic use
  - Exit code 0 if all pass, 1 if any fail
- **Dependencies:** None
- **Estimate:** Medium
- **Risk:** LOW

### Task 4.5: Add `--verbose` and `--quiet` Flags
- **Description:** Global flags for output control
- **Behavior:**
  - `--verbose` / `-v`: Show debug info, HTTP requests, timing
  - `--quiet` / `-q`: Suppress all non-essential output
  - Default: current behavior (moderate output)
- **Acceptance criteria:**
  - Works on all commands
  - `--verbose` shows provider URL, model name, timing
  - `--quiet` only outputs JSON or final result
- **Dependencies:** None
- **Estimate:** Small
- **Risk:** LOW

---

## Phase 5: ACLI Integration

### Task 5.1: Create MCP Adapter
- **Description:** Model Context Protocol adapter for AI agent integration
- **New file:** `openvision/mcp/adapter.py`
- **Design:**
  - Wraps CLI commands as MCP tools
  - JSON input/output via stdin/stdout
  - Tool definitions matching CLI commands
  - Schema-validated inputs/outputs
- **Tool definitions:**
  ```json
  {
    "name": "openvision_observe",
    "description": "Analyze video or image content",
    "inputSchema": {
      "path": "string",
      "question": "string (optional)",
      "mode": "fast|balanced|detailed"
    }
  }
  ```
- **Acceptance criteria:**
  - Can be loaded as MCP server
  - All 8+ commands available as tools
  - Structured JSON I/O
  - Error responses are structured
- **Dependencies:** Phase 2 complete
- **Estimate:** Large
- **Risk:** MEDIUM — MCP spec evolution

### Task 5.2: Create Agent Skill Files
- **Description:** Skill definitions for popular AI frameworks
- **New directory:** `openvision/skills/`
- **Skills to create:**
  - `openvision-observe.md` — for OpenCode/Claude
  - `openvision-analyze.md` — generic agent skill
  - `openvision-webcam.md` — real-time capture skill
  - `openvision-batch.md` — batch processing skill
- **Acceptance criteria:**
  - Skill files follow standard format
  - Include example invocations
  - Document all parameters
  - Tested with at least one AI agent
- **Dependencies:** Phase 2 complete
- **Estimate:** Medium
- **Risk:** LOW

### Task 5.3: Add Webhook Support
- **Description:** Event-driven analysis via webhooks
- **New command:** `openvision serve --port 8000`
- **Design:**
  - HTTP API server (FastAPI or built-in)
  - Endpoints:
    - `POST /analyze` — submit video/image for analysis
    - `GET /status/{job_id}` — check job status
    - `GET /result/{job_id}` — get results
    - `POST /webhook` — register callback URL
  - Async job processing
  - Webhook callback on completion
- **Acceptance criteria:**
  - `openvision serve` starts HTTP server
  - POST to `/analyze` returns job ID
  - Webhook fires on completion
  - Health endpoint available
- **Dependencies:** None
- **Estimate:** Large
- **Risk:** MEDIUM — async job management

### Task 5.4: Add `--output` Flag for File Output
- **Description:** Save results directly to files
- **Behavior:**
  ```bash
  openvision observe video.mp4 --output result.json
  openvision observe video.mp4 --output result.md --format markdown
  ```
- **Acceptance criteria:**
  - `--output <file>` saves result to file
  - Format auto-detected from extension (.json, .md, .txt)
  - `--format` flag for explicit format control
  - Creates parent directories if needed
- **Dependencies:** None
- **Estimate:** Small
- **Risk:** LOW

### Task 5.5: Add `--stdin` Flag for Piped Input
- **Description:** Read paths/questions from stdin for scripting
- **Usage:**
  ```bash
  find . -name "*.jpg" | openvision batch --stdin --json
  echo "video.mp4" | openvision observe --stdin --json
  ```
- **Acceptance criteria:**
  - Reads one path per line from stdin
  - Works with `batch` command
  - Respects `--json` for output
- **Dependencies:** None
- **Estimate:** Small
- **Risk:** LOW

---

## Task Dependencies (Critical Path)

```
Phase 1 (Rename) ─────────────────────────────────────────────┐
  1.1 Rename dir ─┬─→ 1.2 Update imports ─┐                   │
  1.3 Update toml ─┘                      │                   │
  1.4 Update errors ──────────────────────┤                   │
  1.5 Update config/paths ────────────────┤                   │
  1.6 Update README ──────────────────────┤                   │
  1.7 Update .gitignore ──────────────────┤                   │
  1.8 Rename GitHub repo ─────────────────┘──→ Phase 2 start  │
                                                               │
Phase 2 (Multi-Provider) ─────────────────────────────────────┤
  2.1 Provider Registry ──┬──→ 2.2 OllamaProvider             │
                          ├──→ 2.4 OpenAIProvider ──→ 2.3 LlamaCpp │
                          ├──→ 2.5 Refactor LmStudio ──────────┤
                          └──→ 2.6 Config Schema ──→ 2.7 CLI refactor │
                              2.8 Status health checks ←────────┤
                                                               │
Phase 3 (Enhanced Capabilities) ──→ Phase 4 (Self-Hosted) ──→ Phase 5 (ACLI)
  3.1 URL fetching (independent)         4.1 Setup wizard         5.1 MCP adapter
  3.2 Batch processing (independent)     4.2 Model download       5.2 Skill files
  3.3 Webcam/screen (independent)        4.3 Docker               5.3 Webhooks
  3.4 SAM variants (independent)         4.4 Doctor command       5.4 --output flag
  3.5 JSON structure (independent)       5.5 --quiet/--verbose    5.5 --stdin flag
```

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Breaking existing users during rename | HIGH | MEDIUM | Backward compat aliases, migration script |
| Provider auto-detection fails | MEDIUM | MEDIUM | Graceful fallback, clear error messages |
| Docker GPU passthrough issues | MEDIUM | LOW | Test with nvidia-docker, document alternatives |
| MCP spec changes | LOW | HIGH | Pin MCP version, abstract adapter layer |
| Model download failures | MEDIUM | MEDIUM | Retry logic, resume support, local cache |
| Platform-specific issues (screen capture) | MEDIUM | LOW | Feature flags, graceful degradation |
| Performance regression from refactoring | HIGH | LOW | Benchmark before/after, maintain test coverage |

---

## Success Criteria

### Phase 1 (Rename) ✅
- [ ] `pip install -e .` installs as `openvision` (not `ophanim`)
- [ ] `openvision --version` prints "Open Vision 1.0.0"
- [ ] `openvision probe video.mp4` works
- [ ] All existing tests pass with new package name
- [ ] No references to "ophanim" in active code (only deprecation aliases)

### Phase 2 (Multi-Provider) ✅
- [ ] `openvision observe video.mp4 --provider ollama --model llava:13b` works
- [ ] `openvision observe video.mp4 --provider openai --model gpt-4o` works
- [ ] `openvision status` shows all provider health states
- [ ] Auto-detection finds first available provider
- [ ] Provider switching doesn't affect other commands

### Phase 3 (Enhanced Capabilities) ✅
- [ ] `openvision observe "https://example.com/photo.jpg"` works
- [ ] `openvision batch ./photos/ --json` outputs array
- [ ] `openvision capture screen` captures screenshot
- [ ] `openvision doctor` shows all system checks
- [ ] All `--json` outputs are valid, parseable JSON

### Phase 4 (Self-Hosted) ✅
- [ ] `openvision setup` guides through first-time configuration
- [ ] `docker-compose up` starts working stack
- [ ] `openvision doctor` passes all checks
- [ ] Model download works via Ollama integration

### Phase 5 (ACLI Integration) ✅
- [ ] MCP adapter loads and exposes all tools
- [ ] Agent skill files work with OpenCode/Claude
- [ ] `openvision serve` starts HTTP API
- [ ] Webhook callbacks fire on job completion

---

## Implementation Order (Recommended)

**Week 1:** Phase 1 (Rename) — Complete foundation
**Week 2:** Phase 2 (Multi-Provider) — Core new capability
**Week 3:** Phase 3 (Enhanced Capabilities) — Feature expansion
**Week 4:** Phase 4 (Self-Hosted) — User experience
**Week 5:** Phase 5 (ACLI Integration) — Agent ecosystem

**Total estimated effort:** 5-7 weeks (solo developer) or 2-3 weeks (2 developers)

---

## Notes for Other Agents

1. **Backward compatibility is critical:** Many users may have scripts using `ophanim` commands. Keep deprecation aliases for at least one major version.

2. **Test before rename:** Run the full test suite BEFORE starting the rename. Any existing failures will be harder to debug after the rename.

3. **Provider factory pattern:** The registry/factory pattern in Phase 2 is the architectural centerpiece. Get this right and everything else follows.

4. **JSON output is a first-class citizen:** Many AI agents will consume the JSON output. Ensure it's always valid, consistent, and well-documented.

5. **Privacy-first design:** Never send data to external services unless explicitly configured. All providers should default to localhost endpoints.

6. **The `--json` flag is non-negotiable:** Every command must support it. This is how AI agents consume the tool's output.

7. **Environment variable migration:** When renaming env vars, always support the old name as a fallback with a deprecation warning. Don't break existing configurations.

8. **Docker is optional but important:** Many users will want to run this in containers. The Docker setup should work out of the box with GPU passthrough.

9. **MCP adapter should be a thin wrapper:** Don't reimplement logic. Just wrap the CLI commands and parse their JSON output.

10. **Version bump to 1.0.0 signals stability:** Only bump when the core architecture is solid and tested.
