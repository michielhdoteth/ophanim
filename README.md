# Ophanim

**Local vision CLI for agents.**

Ophanim is a local CLI that gives shell-capable agents basic visual perception: video inspection, frame sampling, visual Q&A, transcription, segmentation, tracking, and compact JSON/Markdown outputs.

I built it for my own local agent workflows and open-sourced it in case it helps other people building local agents.

## Quick Start

```bash
pip install -e .
ophanim status
ophanim probe video.mp4
ophanim observe video.mp4 --json
```

See [ophanim/README.md](ophanim/README.md) for full documentation.

## License

MIT
