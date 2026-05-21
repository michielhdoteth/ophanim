# VisionClaw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI tool for video/image perception optimized for RTX 4050 6GB VRAM with LM Studio as VLM backend.

**Architecture:** Python CLI (Typer) with LM Studio HTTP API for vision-language inference, OpenCV for frame extraction, on-demand SAM for segmentation, filesystem+SQLite caching, YAML config.

**Tech Stack:** Python 3.9+, Typer, OpenCV, Pillow, httpx, PyYAML, PyTorch, SAM/ultralytics

---

### Task 1: Project skeleton, dependencies, default config

**Files:**
- Create: `visionclaw/pyproject.toml`
- Create: `visionclaw/__init__.py`
- Create: `visionclaw/__main__.py`
- Create: `visionclaw/config/default.yaml`
- Create: `visionclaw/storage/__init__.py`
- Create: `visionclaw/core/__init__.py`
- Create: `visionclaw/providers/__init__.py`
- Create: `visionclaw/cli/__init__.py`
- Create: `visionclaw/cli/commands/__init__.py`

- [ ] **Step 1: Create pyproject.toml**
- [ ] **Step 2: Create __init__.py files for all packages**
- [ ] **Step 3: Create default.yaml config**
- [ ] **Step 4: Install dependencies**

---

### Task 2: Core video probe + frame extraction

**Files:**
- Create: `visionclaw/core/video.py`
- Create: `visionclaw/core/image.py`
- Test: `visionclaw/tests/test_video.py`

- [ ] **Step 1: Create core/video.py with probe() and extract_frames()**
- [ ] **Step 2: Create core/image.py with downscale, encode, thumbnail**
- [ ] **Step 3: Write and run tests**

---

### Task 3: Smart frame sampling + scene detection

**Files:**
- Create: `visionclaw/core/sampling.py`
- Test: `visionclaw/tests/test_sampling.py`

- [ ] **Step 1: Create core/sampling.py with scene detection, dedup, smart_sample**
- [ ] **Step 2: Write and run tests**

---

### Task 4: LM Studio VLM provider

**Files:**
- Create: `visionclaw/providers/base.py`
- Create: `visionclaw/providers/lmstudio.py`
- Create: `visionclaw/models.py`
- Test: `visionclaw/tests/test_providers.py`

- [ ] **Step 1: Create models.py with pydantic schemas**
- [ ] **Step 2: Create providers/base.py with abstract VlmProvider**
- [ ] **Step 3: Create providers/lmstudio.py with LM Studio client**
- [ ] **Step 4: Write and run tests**

---

### Task 5: CLI entry point + probe command

**Files:**
- Create: `visionclaw/cli/app.py`
- Create: `visionclaw/cli/commands/probe.py`
- Test: `visionclaw/tests/test_cli.py`

- [ ] **Step 1: Create cli/app.py with Typer app**
- [ ] **Step 2: Create cli/commands/probe.py**
- [ ] **Step 3: Write and run tests**

---

### Task 6: observe command

**Files:**
- Create: `visionclaw/cli/commands/observe.py`
- Test: `visionclaw/tests/test_observe.py`

- [ ] **Step 1: Create cli/commands/observe.py**
- [ ] **Step 2: Write and run tests**

---

### Task 7: ask command

**Files:**
- Create: `visionclaw/cli/commands/ask.py`
- Test: `visionclaw/tests/test_ask.py`

- [ ] **Step 1: Create cli/commands/ask.py**
- [ ] **Step 2: Write and run tests**

---

### Task 8: segment + track commands with SAM

**Files:**
- Create: `visionclaw/providers/sam.py`
- Create: `visionclaw/cli/commands/segment.py`
- Create: `visionclaw/cli/commands/track.py`
- Test: `visionclaw/tests/test_segment.py`

- [ ] **Step 1: Create providers/sam.py**
- [ ] **Step 2: Create cli/commands/segment.py**
- [ ] **Step 3: Create cli/commands/track.py**
- [ ] **Step 4: Write and run tests**

---

### Task 9: GPU memory manager

**Files:**
- Create: `visionclaw/core/gpu.py`
- Test: `visionclaw/tests/test_gpu.py`

- [ ] **Step 1: Create core/gpu.py with VRAM detection, safe mode, auto-downgrade**
- [ ] **Step 2: Write and run tests**

---

### Task 10: Status command + run caching + config

**Files:**
- Create: `visionclaw/storage/cache.py`
- Create: `visionclaw/storage/config.py`
- Create: `visionclaw/cli/commands/status.py`
- Test: `visionclaw/tests/test_cache.py`

- [ ] **Step 1: Create storage/config.py**
- [ ] **Step 2: Create storage/cache.py**
- [ ] **Step 3: Create cli/commands/status.py**
- [ ] **Step 4: Write and run tests**

---

### Task 11: Error handling system

**Files:**
- Modify: all command files
- Create: `visionclaw/core/errors.py`
- Test: `visionclaw/tests/test_errors.py`

- [ ] **Step 1: Create core/errors.py with error types**
- [ ] **Step 2: Integrate errors into all commands**
- [ ] **Step 3: Write and run tests**

---

### Task 12: Memory output, README, examples

**Files:**
- Create: `visionclaw/cli/commands/memory.py`
- Create: `visionclaw/README.md`
- Create: `visionclaw/EXAMPLES.md`

- [ ] **Step 1: Create memory.py for markdown output**
- [ ] **Step 2: Create README.md**
- [ ] **Step 3: Create EXAMPLES.md**
- [ ] **Step 4: Final review and polish**
