# Ophanim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI tool for video/image perception optimized for RTX 4050 6GB VRAM with LM Studio as VLM backend.

**Architecture:** Python CLI (Typer) with LM Studio HTTP API for vision-language inference, OpenCV for frame extraction, on-demand SAM for segmentation, filesystem+SQLite caching, YAML config.

**Tech Stack:** Python 3.9+, Typer, OpenCV, Pillow, httpx, PyYAML, PyTorch, SAM/ultralytics

---

### Task 1: Project skeleton, dependencies, default config

**Files:**
- Create: `ophanim/pyproject.toml`
- Create: `ophanim/__init__.py`
- Create: `ophanim/__main__.py`
- Create: `ophanim/config/default.yaml`
- Create: `ophanim/storage/__init__.py`
- Create: `ophanim/core/__init__.py`
- Create: `ophanim/providers/__init__.py`
- Create: `ophanim/cli/__init__.py`
- Create: `ophanim/cli/commands/__init__.py`

- [ ] **Step 1: Create pyproject.toml**
- [ ] **Step 2: Create __init__.py files for all packages**
- [ ] **Step 3: Create default.yaml config**
- [ ] **Step 4: Install dependencies**

---

### Task 2: Core video probe + frame extraction

**Files:**
- Create: `ophanim/core/video.py`
- Create: `ophanim/core/image.py`
- Test: `ophanim/tests/test_video.py`

- [ ] **Step 1: Create core/video.py with probe() and extract_frames()**
- [ ] **Step 2: Create core/image.py with downscale, encode, thumbnail**
- [ ] **Step 3: Write and run tests**

---

### Task 3: Smart frame sampling + scene detection

**Files:**
- Create: `ophanim/core/sampling.py`
- Test: `ophanim/tests/test_sampling.py`

- [ ] **Step 1: Create core/sampling.py with scene detection, dedup, smart_sample**
- [ ] **Step 2: Write and run tests**

---

### Task 4: LM Studio VLM provider

**Files:**
- Create: `ophanim/providers/base.py`
- Create: `ophanim/providers/lmstudio.py`
- Create: `ophanim/models.py`
- Test: `ophanim/tests/test_providers.py`

- [ ] **Step 1: Create models.py with pydantic schemas**
- [ ] **Step 2: Create providers/base.py with abstract VlmProvider**
- [ ] **Step 3: Create providers/lmstudio.py with LM Studio client**
- [ ] **Step 4: Write and run tests**

---

### Task 5: CLI entry point + probe command

**Files:**
- Create: `ophanim/cli/app.py`
- Create: `ophanim/cli/commands/probe.py`
- Test: `ophanim/tests/test_cli.py`

- [ ] **Step 1: Create cli/app.py with Typer app**
- [ ] **Step 2: Create cli/commands/probe.py**
- [ ] **Step 3: Write and run tests**

---

### Task 6: observe command

**Files:**
- Create: `ophanim/cli/commands/observe.py`
- Test: `ophanim/tests/test_observe.py`

- [ ] **Step 1: Create cli/commands/observe.py**
- [ ] **Step 2: Write and run tests**

---

### Task 7: ask command

**Files:**
- Create: `ophanim/cli/commands/ask.py`
- Test: `ophanim/tests/test_ask.py`

- [ ] **Step 1: Create cli/commands/ask.py**
- [ ] **Step 2: Write and run tests**

---

### Task 8: segment + track commands with SAM

**Files:**
- Create: `ophanim/providers/sam.py`
- Create: `ophanim/cli/commands/segment.py`
- Create: `ophanim/cli/commands/track.py`
- Test: `ophanim/tests/test_segment.py`

- [ ] **Step 1: Create providers/sam.py**
- [ ] **Step 2: Create cli/commands/segment.py**
- [ ] **Step 3: Create cli/commands/track.py**
- [ ] **Step 4: Write and run tests**

---

### Task 9: GPU memory manager

**Files:**
- Create: `ophanim/core/gpu.py`
- Test: `ophanim/tests/test_gpu.py`

- [ ] **Step 1: Create core/gpu.py with VRAM detection, safe mode, auto-downgrade**
- [ ] **Step 2: Write and run tests**

---

### Task 10: Status command + run caching + config

**Files:**
- Create: `ophanim/storage/cache.py`
- Create: `ophanim/storage/config.py`
- Create: `ophanim/cli/commands/status.py`
- Test: `ophanim/tests/test_cache.py`

- [ ] **Step 1: Create storage/config.py**
- [ ] **Step 2: Create storage/cache.py**
- [ ] **Step 3: Create cli/commands/status.py**
- [ ] **Step 4: Write and run tests**

---

### Task 11: Error handling system

**Files:**
- Modify: all command files
- Create: `ophanim/core/errors.py`
- Test: `ophanim/tests/test_errors.py`

- [ ] **Step 1: Create core/errors.py with error types**
- [ ] **Step 2: Integrate errors into all commands**
- [ ] **Step 3: Write and run tests**

---

### Task 12: Memory output, README, examples

**Files:**
- Create: `ophanim/cli/commands/memory.py`
- Create: `ophanim/README.md`
- Create: `ophanim/EXAMPLES.md`

- [ ] **Step 1: Create memory.py for markdown output**
- [ ] **Step 2: Create README.md**
- [ ] **Step 3: Create EXAMPLES.md**
- [ ] **Step 4: Final review and polish**
