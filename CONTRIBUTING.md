# Contributing to Open Vision

Thanks for your interest in contributing. Open Vision is a local CLI tool that gives AI agents visual perception. Contributions are welcome.

## Getting Started

```bash
git clone https://github.com/michielhdoteth/openvision.git
cd openvision
pip install -e ".[test]"
pytest tests/ -v
```

You need Python 3.12+, ffmpeg, and optionally a GPU for VLM inference.

## What to Work On

### Good first issues

- [Bug reports](https://github.com/michielhdoteth/openvision/issues?q=is%3Aissue+is%3Aopen+label%3Abug) -- something broken
- [Documentation](https://github.com/michielhdoteth/openvision/issues?q=is%3Aissue+is%3Aopen+label%3Adocs) -- missing or unclear docs
- [Good first issue](https://github.com/michielhdoteth/openvision/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) -- beginner-friendly

### Harder to review

- New features -- adds surface area and maintenance burden
- Refactoring -- hard to verify no behavior change
- New dependencies -- adds weight, needs strong justification

### May not be accepted

- Breaking API changes
- Changes that add friction to the CLI UX
- Changes that create large maintenance burden without clear benefit

## Proposing a Change

For non-trivial changes, **open an issue first** for discussion.

Good proposals:
- Explain the **problem**, not what you want to do
- Explain **why** the change matters
- Explain **how** it will be used
- Explain **how** it will be tested

## Pull Requests

### Commit messages

Use this format:

```
<area>: <short description>
```

Examples:
```
core: add CUDA transpose for GPU frame extraction
providers: fix parakeet model loading on Windows
cli: add --jsonl streaming output to observe command
tests: add unit tests for SQLite cache
docs: update README with new features
```

### What we look for

- Tests for new functionality
- No regressions (existing tests pass)
- Clean, focused diffs (one thing per PR)
- Updated docs if the user-facing behavior changes

### Running tests

```bash
pytest tests/ -v                              # All tests
pytest tests/ -m "not slow" -v                # Skip slow tests
pytest tests/test_stream.py -v                # Single module
pytest tests/ --cov=core --cov=providers      # With coverage
```

## Code Style

- Python 3.12+ (use modern syntax: `list[dict]`, `X | None`, `match`)
- Type hints on public functions
- No comments unless the logic is genuinely non-obvious
- Follow existing patterns in the codebase

## Security

If you discover a security vulnerability, do not open a public issue. See [SECURITY.md](SECURITY.md) for reporting instructions.
