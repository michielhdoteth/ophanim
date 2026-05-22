"""ophanim observe <path> - Observe video or image."""
import typer
import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from ophanim.core.video import probe, extract_frames, estimate_processing_cost
from ophanim.core.image import downscale, encode_base64, save_frame, load_image
from ophanim.core.sampling import smart_sample
from ophanim.core.gpu import auto_downgrade_mode, log_vram
from ophanim.providers.lmstudio import LmStudioProvider
from ophanim.storage.cache import RunCache
from ophanim.storage.config import load_config, get_mode_config
from ophanim.models import (
    ObserveResult, ImageResult, TimelineEntry, ProbeResult,
)

console = Console()


def observe_cmd(
    path: str = typer.Argument(..., help="Path to image or video file"),
    question: str = typer.Option(None, "--question", "-q", help="Specific question to answer"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Processing mode: fast, balanced, detailed"),
    fps: float = typer.Option(None, "--fps", help="Frames per second to sample"),
    max_frames: int = typer.Option(None, "--max-frames", help="Maximum frames to process"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    save_memory: bool = typer.Option(False, "--save-memory", help="Save observation as markdown memory"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocess (ignore cache)"),
):
    """Analyze a video or image and return observations."""
    # Validate file exists
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    # Check GPU and potentially downgrade
    actual_mode, warning = auto_downgrade_mode(mode, config)
    if warning:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if actual_mode != mode:
        mode = actual_mode
        mode_config = get_mode_config(config, mode)

    # Determine final params
    actual_fps = fps or mode_config.get("fps", 0.5)
    actual_max_frames = max_frames or mode_config.get("max_frames", 60)
    actual_resolution = mode_config.get("resolution", 768)

    # Check if it's an image
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
    is_image = input_path.suffix.lower() in image_extensions

    if is_image:
        _handle_image(input_path, question, json_output, config, actual_resolution)
    else:
        _handle_video(input_path, question, json_output, save_memory, force,
                      config, mode, actual_fps, actual_max_frames, actual_resolution)


def _handle_image(path: Path, question: Optional[str], json_output: bool,
                  config: dict, resolution: int):
    """Process a single image."""
    # Load and preprocess
    image = load_image(str(path))
    image = downscale(image, resolution)

    # Query LM Studio
    vlm_config = config.get("models", {}).get("vlm", {})
    provider = LmStudioProvider(vlm_config)

    if not provider.check_health():
        console.print("[red]Error: LM Studio is not running.[/red]")
        console.print("Start LM Studio and load google/gemma-4-e2b, then try again.")
        raise typer.Exit(code=1)

    try:
        description = provider.describe_image(image, question)
    except Exception as e:
        console.print(f"[red]Error querying VLM:[/red] {e}")
        raise typer.Exit(code=1)
    finally:
        provider.close()

    result = ImageResult(
        description=description,
        objects=[],  # Optional: can extract from description
        text_detected=[],
        confidence="medium",
    )

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        console.print(Panel(description, title="[bold cyan]Image Observation[/bold cyan]"))


def _handle_video(path: Path, question: Optional[str], json_output: bool,
                  save_memory: bool, force: bool, config: dict, mode: str,
                  fps: float, max_frames: int, resolution: int):
    """Process a video file."""
    # Check cache
    cache = RunCache(config.get("cache", {}).get("directory", "./runs"))
    key = cache.cache_key(str(path), mode, fps, resolution)

    if not force and key and cache.has_cached(key):
        cached_dir = cache.get_run(key)
        cached_file = cached_dir / "observations.json" if cached_dir else None
        if cached_file and cached_file.exists():
            import json as j
            data = j.loads(cached_file.read_text())
            if json_output:
                console.print(j.dumps(data, indent=2))
            else:
                console.print("[green]Using cached result.[/green]")
                _display_observation(data)
            return

    # Probe video
    log_vram("video_probe")
    video_meta = probe(str(path))

    # Estimate cost
    cost = estimate_processing_cost(video_meta, mode)
    if cost == "high":
        console.print(f"[yellow]Video is long ({video_meta['duration_seconds']:.0f}s). "
                      f"This may take a while.[/yellow]")

    # Sample frames
    log_vram("frame_sampling")
    frames = smart_sample(
        str(path),
        fps=fps,
        max_frames=max_frames,
        max_resolution=resolution,
    )

    if not frames:
        console.print("[red]No frames extracted from video.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Extracted {len(frames)} frames for analysis[/dim]")

    # Create run directory
    run_dir = cache.create_run(key, video_meta)

    # Save frames
    frame_paths = []
    for i, frame in enumerate(frames):
        fname = f"frame_{i:04d}.jpg"
        fpath = cache.save_frame(run_dir, frame["image"], fname)
        frame_paths.append(str(fpath))
        frame["path"] = str(fpath)

    # Query VLM
    vlm_config = config.get("models", {}).get("vlm", {})
    provider = LmStudioProvider(vlm_config)

    if not provider.check_health():
        console.print("[red]Error: LM Studio is not running.[/red]")
        console.print("Start LM Studio and load google/gemma-4-e2b, then try again.")
        raise typer.Exit(code=1)

    log_vram("vlm_observation")

    # Process frames with VLM
    timeline = []
    entities = set()

    for i, frame in enumerate(frames):
        if question:
            result_text = provider.describe_image(frame["image"], question)
        else:
            result_text = provider.describe_image(
                frame["image"],
                "Describe what is happening in this frame. Focus on objects, people, actions. Be concise (1-2 sentences)."
            )

        timeline.append(TimelineEntry(
            time_seconds=frame["timestamp"],
            timestamp=frame["timestamp_str"],
            observation=result_text,
            frame_path=frame_paths[i] if i < len(frame_paths) else None,
        ))

        # Extract simple entities from description
        # (basic keyword extraction; could be improved later)
        for word in result_text.split():
            clean = word.strip(".,!?:;\"'()[]").lower()
            if clean and len(clean) > 3 and clean.isalpha():
                if clean not in ("this", "that", "with", "from", "what", "there", "they", "when", "which"):
                    entities.add(clean)

    provider.close()

    # Generate summary
    if question:
        summary = _build_qa_summary(timeline, question)
    else:
        summary = _build_timeline_summary(timeline)

    # Build result
    result = ObserveResult(
        summary=summary,
        timeline=timeline,
        entities=sorted(list(entities))[:20],  # Cap entities
        artifacts_dir=str(run_dir),
        confidence="medium",
    )

    # Save artifacts
    cache.save_artifact(run_dir, "observations.json", result.model_dump())
    cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")

    # Timeline markdown
    timeline_md = "# Timeline\n\n"
    for entry in timeline:
        timeline_md += f"- **{entry.timestamp}** - {entry.observation}\n"
    cache.save_text(run_dir, "timeline.md", timeline_md)

    # Save memory markdown if requested
    if save_memory:
        _save_memory_md(path, result, config)

    log_vram("observation_complete")

    # Output
    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        _display_observation(result.model_dump())


def _build_timeline_summary(timeline: list) -> str:
    """Generate a summary from timeline entries."""
    if not timeline:
        return "No observations recorded."

    observations = [t.observation for t in timeline if t.observation]
    if not observations:
        return "No observations recorded."

    # Use first and last observations to frame summary
    first = observations[0]
    last = observations[-1]

    if len(observations) == 1:
        return first

    return f"The video shows: {first} Towards the end, {last}"


def _build_qa_summary(timeline: list, question: str) -> str:
    """Compile answers to a specific question across timeline."""
    if not timeline:
        return "No frames available to answer the question."

    answers = [t.observation for t in timeline if t.observation and len(t.observation) > 5]

    if not answers:
        return "Could not determine answer from video frames."

    # Take the most detailed answer
    best = max(answers, key=len)
    return f"Based on video analysis: {best}"


def _display_observation(data: dict):
    """Display observation result in rich format."""
    console.print(Panel(data.get("summary", ""), title="[bold cyan]Observation Summary[/bold cyan]"))

    if data.get("timeline"):
        console.print("\n[bold]Timeline:[/bold]")
        for entry in data["timeline"][:10]:  # Show first 10
            obs = entry.get("observation", "")
            ts = entry.get("timestamp", "??:??")
            if len(obs) > 100:
                obs = obs[:100] + "..."
            console.print(f"  [{ts}] {obs}")
        if len(data["timeline"]) > 10:
            console.print(f"  [dim]... and {len(data['timeline']) - 10} more entries[/dim]")

    if data.get("entities"):
        console.print(f"\n[bold]Entities:[/bold] {', '.join(data['entities'][:10])}")

    if data.get("artifacts_dir"):
        console.print(f"\n[dim]Artifacts: {data['artifacts_dir']}[/dim]")


def _save_memory_md(path: Path, result: ObserveResult, config: dict):
    """Save observation as markdown memory file."""
    from datetime import date
    today = date.today().isoformat()
    video_name = path.stem

    # Create memory directory
    memory_dir = Path("memory") / "videos"
    memory_dir.mkdir(parents=True, exist_ok=True)

    memory_path = memory_dir / f"{today}-{video_name}.md"
    lines = [
        f"# Video Observation: {path.name}",
        "",
        f"Date: {today}",
        f"Duration: {result.artifacts_dir}",
        "",
        "## Summary",
        "",
        result.summary,
        "",
        "## Timeline",
        "",
    ]
    for entry in result.timeline:
        lines.append(f"- **{entry.timestamp}** - {entry.observation}")

    if result.entities:
        lines.extend(["", "## Entities", ""])
        for entity in result.entities:
            lines.append(f"- {entity}")

    lines.extend(["", "## Artifacts", "", f"- Frames: `{result.artifacts_dir}/frames/`"])

    memory_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Memory saved:[/green] {memory_path}")
