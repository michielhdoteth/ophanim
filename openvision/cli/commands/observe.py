"""openvision observe <path> - Observe video or image."""
import tempfile
import typer
import json
import os
import cv2
from pathlib import Path
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn

from openvision.core.video import probe, extract_frames, estimate_processing_cost, auto_fps, _downscale
from openvision.core.image import downscale, encode_base64, save_frame, load_image
from openvision.core.sampling import smart_sample
from openvision.core.gpu import auto_downgrade_mode, log_vram
from openvision.providers.registry import ProviderRegistry
from openvision.providers.whisper import WhisperProvider, Transcript
from openvision.providers.base import VlmResponse, TokenUsage
from openvision.storage.cache import RunCache
from openvision.storage.config import load_config, get_mode_config
from openvision.models import (
    ObserveResult, ImageResult, TimelineEntry, ProbeResult,
)

console = Console()


def _fmt_time(seconds: float) -> str:
    """Format seconds to MM:SS."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_tokens(tokens: TokenUsage) -> str:
    """Format token usage for display."""
    parts = [f"{tokens.prompt_tokens} prompt"]
    if tokens.reasoning_tokens > 0:
        parts.append(f"{tokens.completion_tokens} completion ({tokens.reasoning_tokens} reasoning)")
    else:
        parts.append(f"{tokens.completion_tokens} completion")
    return f"{tokens.total_tokens} total | " + " + ".join(parts)


class _EntityResult:
    """Helper to carry entities + tokens from _extract_entities."""
    def __init__(self, entities: list[str], tokens: TokenUsage):
        self.entities = entities
        self.tokens = tokens


def observe_cmd(
    path: str = typer.Argument(..., help="Path to image or video file"),
    question: str = typer.Option(None, "--question", "-q", help="Specific question to answer (hardcoded prompt)"),
    prompt: str = typer.Option(None, "--prompt", "-p", help="Entirely custom prompt (overrides default)"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Processing mode: fast, balanced, detailed"),
    detail: str = typer.Option("balanced", "--detail", help="Extraction detail level: transcript, efficient, balanced, token-burner"),
    fps: float = typer.Option(None, "--fps", help="Frames per second to sample"),
    max_frames: int = typer.Option(None, "--max-frames", help="Maximum frames to process"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Override max tokens for VLM response"),
    device: str = typer.Option("auto", "--device", "-d", help="Whisper device: auto, cpu, cuda"),
    start_time: str = typer.Option(None, "--start", help="Start time for focus range (e.g., 1:30, 45, 0:15:00)"),
    end_time: str = typer.Option(None, "--end", help="End time for focus range (e.g., 2:00, 90, 0:20:00)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    save_memory: bool = typer.Option(False, "--save-memory", help="Save observation as markdown memory"),
    transcribe_audio: bool = typer.Option(False, "--transcribe", "-t", help="Transcribe audio speech to text"),
    diarize_audio: bool = typer.Option(False, "--diarize", help="Add speaker labels via diarization"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocess (ignore cache)"),
    timestamps: str = typer.Option(None, "--timestamps", help="Extract frames at specific timestamps (comma-separated, e.g., '0:30,2:15,5:00')"),
    provider_name: str = typer.Option(None, "--provider", help="VLM provider: auto, lmstudio, ollama, llamacpp, openai, groq, together, vllm, localai"),
):
    """Analyze a video or image and return observations."""
    # Validate file exists or is URL
    from openvision.core.download import is_url
    input_path = Path(path)
    is_remote = is_url(path)
    if not is_remote and not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    # Download from URL if remote (native yt-dlp into openvision_HOME/downloads)
    if is_remote:
        from openvision.core.download import download_video
        from openvision.storage.paths import downloads_dir
        dl_dir = downloads_dir(config)
        console.print(f"[dim]Downloading from URL into {dl_dir}...[/dim]")
        try:
            dl_result = download_video(path, output_dir=str(dl_dir), max_height=720)
            input_path = Path(dl_result["path"])
            dur = dl_result.get("duration") or 0
            console.print(
                f"[dim]Downloaded: {dl_result.get('title', input_path.name)} "
                f"({dur:.0f}s) -> {input_path}[/dim]"
            )
            if dl_result.get("subs_file"):
                console.print(f"[dim]Subtitles: {dl_result['subs_file']}[/dim]")
        except Exception as e:
            console.print(f"[red]Download failed:[/red] {e}")
            raise typer.Exit(code=1)

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

    # Parse focus range
    focus_start = None
    focus_end = None
    if start_time:
        from openvision.core.video import parse_time
        focus_start = parse_time(start_time)
    if end_time:
        from openvision.core.video import parse_time
        focus_end = parse_time(end_time)
    has_focus = focus_start is not None or focus_end is not None

    # Parse cue timestamps
    cue_timestamps = None
    if timestamps:
        from openvision.core.video import parse_time
        try:
            cue_timestamps = [parse_time(t.strip()) for t in timestamps.split(",")]
        except ValueError as e:
            console.print(f"[red]Invalid timestamp: {e}[/red]")
            raise typer.Exit(code=1)

    # Check if it's an image
    if is_remote:
        is_image = False
    else:
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
        is_image = input_path.suffix.lower() in image_extensions

    if is_image:
        _handle_image(input_path, question, prompt, json_output, config, actual_resolution, max_tokens, provider_name)
    else:
        _handle_video(input_path, question, prompt, json_output, save_memory, transcribe_audio, diarize_audio, force,
                      config, mode, detail, actual_fps, actual_max_frames, actual_resolution, max_tokens, device,
                      focus_start, focus_end, cue_timestamps, provider_name)


def _handle_image(path: Path, question: Optional[str], custom_prompt: Optional[str],
                  json_output: bool, config: dict, resolution: int,
                  override_max_tokens: Optional[int] = None,
                  provider_name: Optional[str] = None):
    """Process a single image."""
    # Load and preprocess
    image = load_image(str(path))
    image = downscale(image, resolution)

    # Create VLM provider via registry
    vlm_config = config.get("models", {}).get("vlm", {})
    if override_max_tokens:
        vlm_config = {**vlm_config, "max_tokens": override_max_tokens}
    if provider_name:
        vlm_config["provider"] = provider_name

    try:
        provider = ProviderRegistry.create(vlm_config)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

    if not provider.check_health():
        provider_name_display = vlm_config.get("provider", "auto")
        console.print(f"[red]Error: {provider_name_display} is not running or not reachable.[/red]")
        console.print("Start your VLM provider and load a vision model, then try again.")
        raise typer.Exit(code=1)

    # Determine prompt: custom_prompt > question > default
    effective_prompt = None
    if custom_prompt:
        effective_prompt = custom_prompt
    elif question:
        effective_prompt = f"Answer this question about the image: {question}\nBe concise and specific."

    try:
        response = provider.describe_image(image, effective_prompt)
        description = response.content
        tokens = response.tokens
    except Exception as e:
        console.print(f"[red]Error querying VLM:[/red] {e}")
        raise typer.Exit(code=1)
    finally:
        provider.close()

    result = ImageResult(
        description=description,
        objects=[],
        text_detected=[],
        confidence="medium",
        tokens={"prompt_tokens": tokens.prompt_tokens, "completion_tokens": tokens.completion_tokens, "reasoning_tokens": tokens.reasoning_tokens, "total_tokens": tokens.total_tokens},
    )

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        console.print(Panel(description, title="[bold cyan]Image Observation[/bold cyan]"))
        console.print(f"[dim]Tokens: {_fmt_tokens(tokens)}[/dim]")


def _extract_focus_range(path: str, start: Optional[float], end: Optional[float],
                         max_frames: int, resolution: int) -> list[dict]:
    """Extract frames from a specific time range using ffmpeg seek."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["ffmpeg", "-i", path]
        if start is not None:
            cmd.extend(["-ss", str(start)])
        if end is not None:
            cmd.extend(["-to", str(end)])

        # Calculate fps to stay within budget
        duration = (end or 0) - (start or 0)
        if duration > 0 and max_frames > 0:
            target_fps = min(6.0, max_frames / duration)
        else:
            target_fps = 1.0

        cmd.extend([
            "-vf", f"fps={target_fps:.2f}",
            "-vsync", "vfr",
            os.path.join(tmpdir, "frame_%06d.jpg"),
        ])

        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            console.print("[yellow]Warning: focus range extraction timed out[/yellow]")
            return []

        frames = []
        for i, fpath in enumerate(sorted(Path(tmpdir).glob("frame_*.jpg"))):
            if i >= max_frames:
                break
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            img = _downscale(img, resolution)
            # Calculate timestamp from index
            ts = (start or 0) + i * (duration / max(1, len(list(Path(tmpdir).glob("frame_*.jpg")))))
            frames.append({
                "index": i,
                "timestamp": ts,
                "timestamp_str": _fmt_time(ts),
                "image": img,
                "reason": "focus-range",
            })

        return frames


def _handle_video(path: Path, question: Optional[str], custom_prompt: Optional[str],
                  json_output: bool, save_memory: bool, transcribe_audio: bool, diarize_audio: bool, force: bool,
                  config: dict, mode: str, detail: str = "balanced",
                  fps: float = 0.5, max_frames: int = 60, resolution: int = 768,
                  override_max_tokens: Optional[int] = None,
                  whisper_device: str = "auto",
                  focus_start: Optional[float] = None,
                  focus_end: Optional[float] = None,
                  cue_timestamps: Optional[list[float]] = None,
                  provider_name: Optional[str] = None):
    """Process a video file."""
    # Check cache (stable under openvision_HOME when relative)
    cache_dir = config.get("cache", {}).get("directory", "runs")
    if not Path(cache_dir).is_absolute():
        from openvision.storage.paths import runs_dir
        cache_dir = str(runs_dir(config))
    cache = RunCache(cache_dir)
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

    # Smart frame budget based on duration
    duration = video_meta["duration_seconds"]
    auto_max = auto_fps(duration)
    actual_max = min(max_frames, auto_max)
    console.print(f"[dim]Video: {duration:.0f}s, budget: {actual_max} frames[/dim]")

    # Estimate cost
    has_focus = focus_start is not None or focus_end is not None
    if not has_focus:
        cost = estimate_processing_cost(video_meta, mode)
        if cost == "high":
            console.print(f"[yellow]Video is long ({duration:.0f}s). "
                          f"Use --start/--end to focus on a specific section.[/yellow]")

    # Sample frames based on detail mode
    log_vram("frame_sampling")
    has_focus = focus_start is not None or focus_end is not None

    if has_focus:
        from openvision.core.video import auto_fps_focus
        focus_duration = (focus_end or duration) - (focus_start or 0)
        focus_max = auto_fps_focus(focus_duration)
        console.print(f"[dim]Focus range: {_fmt_time(focus_start or 0)} - {_fmt_time(focus_end or duration)} ({focus_duration:.0f}s, budget: {focus_max} frames)[/dim]")
        frames = _extract_focus_range(str(path), focus_start, focus_end, focus_max, resolution)
    elif detail == "transcript":
        console.print("[dim]Transcript-only mode: skipping frame extraction[/dim]")
        frames = []
    elif detail == "efficient":
        from openvision.core.video import extract_keyframes_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Efficient mode: extracting keyframes (cap 50)[/dim]")
        frames = extract_keyframes_ffmpeg(str(path), max_frames=50, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    elif detail == "token-burner":
        from openvision.core.video import extract_scene_frames_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Token-burner mode: full scene detection (uncapped)[/dim]")
        frames = extract_scene_frames_ffmpeg(str(path), max_frames=9999, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    else:
        # balanced (default)
        if duration > 120:
            from openvision.core.sampling import adaptive_sample
            frames = adaptive_sample(str(path), max_frames=actual_max, max_resolution=resolution)
        else:
            frames = smart_sample(str(path), fps=fps, max_frames=actual_max, max_resolution=resolution)

    # Transcript-only early return: skip VLM, just run audio processing
    if detail == "transcript" and not frames:
        run_dir = cache.create_run(key, video_meta)
        transcript = None
        timeline = []

        if transcribe_audio:
            console.print(f"[dim]Transcribing audio with Whisper ({whisper_device})...[/dim]")
            whisper = WhisperProvider({"device": whisper_device})
            transcript = whisper.transcribe(str(path))
            if transcript and transcript.segments:
                console.print(f"[dim]Transcribed {len(transcript.segments)} speech segments[/dim]")

        if diarize_audio and transcript and transcript.segments:
            try:
                from openvision.providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
                from openvision.core.audio import extract_audio

                console.print("[dim]Running speaker diarization...[/dim]")
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp_path = tmp.name
                tmp.close()
                try:
                    extract_audio(str(path), output_path=tmp_path)
                    diarizer = DiarizerProvider()
                    dia_result = diarizer.diarize(tmp_path)
                    transcript.segments = merge_transcript_with_diarization(
                        transcript.segments, dia_result.segments
                    )
                    console.print(f"[dim]Found {dia_result.num_speakers} speakers: {', '.join(dia_result.speakers)}[/dim]")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
            except ImportError as e:
                console.print(f"[yellow]Diarization unavailable: {e}[/yellow]")
            except Exception as e:
                console.print(f"[yellow]Diarization failed: {e}[/yellow]")

        if transcript and transcript.segments:
            for seg in transcript.segments:
                ts = _fmt_time(seg.start)
                speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
                timeline.append(TimelineEntry(
                    time_seconds=seg.start,
                    timestamp=ts,
                    observation=f"[SPEECH] {speaker_prefix}{seg.text}",
                    frame_path=None,
                ))

        # Build minimal summary
        if timeline:
            summary = f"Transcript-only analysis: {len(transcript.segments)} speech segments transcribed."
        else:
            summary = "Transcript-only mode: no audio transcribed."

        result = ObserveResult(
            summary=summary,
            timeline=timeline,
            entities=[],
            artifacts_dir=str(run_dir),
            confidence="medium",
            tokens={"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0},
        )

        cache.save_artifact(run_dir, "observations.json", result.model_dump())
        cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")

        timeline_md = "# Timeline\n\n"
        for entry in timeline:
            timeline_md += f"- **{entry.timestamp}** - {entry.observation}\n"
        cache.save_text(run_dir, "timeline.md", timeline_md)

        if transcript and transcript.segments:
            transcript_text = "\n".join(
                f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
                for s in transcript.segments
            )
            cache.save_text(run_dir, "transcript.txt", transcript_text)

        if save_memory:
            _save_memory_md(path, result, config, transcript)

        log_vram("observation_complete")

        if json_output:
            console.print(json.dumps(result.model_dump(), indent=2))
        else:
            _display_observation(result.model_dump())
        return

    if not frames:
        console.print("[red]No frames extracted from video.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Extracted {len(frames)} frames for analysis[/dim]")

    # Extract pinned frames at specific timestamps
    pinned_frames = []
    if cue_timestamps:
        from openvision.core.video import extract_at_timestamps
        pinned_frames = extract_at_timestamps(str(path), cue_timestamps, resolution)

    # Merge pinned frames into main frame list (dedup by timestamp)
    if pinned_frames:
        existing_times = {round(f["timestamp"], 1) for f in frames}
        for pf in pinned_frames:
            if round(pf["timestamp"], 1) not in existing_times:
                frames.append(pf)
                existing_times.add(round(pf["timestamp"], 1))
        frames.sort(key=lambda f: f["timestamp"])

    # Warn about sparse scan for long videos
    if duration > 600 and len(frames) < 30:
        console.print(f"[yellow]Warning: {duration/60:.0f}-minute video with only {len(frames)} frames. "
                      f"Consider --start/--end to focus, or --detail token-burner for full scene detection.[/yellow]")

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
    if override_max_tokens:
        vlm_config = {**vlm_config, "max_tokens": override_max_tokens}
    if provider_name:
        vlm_config["provider"] = provider_name

    try:
        provider = ProviderRegistry.create(vlm_config)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(code=1)

    if not provider.check_health():
        provider_name_display = vlm_config.get("provider", "auto")
        console.print(f"[red]Error: {provider_name_display} is not running or not reachable.[/red]")
        console.print("Start your VLM provider and load a vision model, then try again.")
        raise typer.Exit(code=1)

    log_vram("vlm_observation")

    # Process frames with VLM
    timeline = []
    total_tokens = TokenUsage()

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            description=f"Analyzing {len(frames)} frames with VLM...",
            total=len(frames),
        )

        for i, frame in enumerate(frames):
            progress.update(task, advance=1, description=f"Frame {i+1}/{len(frames)} at {frame['timestamp_str']}")

            # Determine prompt: custom_prompt > question > default
            if custom_prompt:
                resp = provider.describe_image(frame["image"], custom_prompt)
            elif question:
                resp = provider.describe_image(frame["image"], f"Answer this question about the image: {question}\nBe concise and specific.")
            else:
                resp = provider.describe_image(
                    frame["image"],
                    "Describe what is happening in this frame. Focus on objects, people, actions. Be concise (1-2 sentences)."
                )

            # Accumulate tokens
            total_tokens.prompt_tokens += resp.tokens.prompt_tokens
            total_tokens.completion_tokens += resp.tokens.completion_tokens
            total_tokens.reasoning_tokens += resp.tokens.reasoning_tokens
            total_tokens.total_tokens += resp.tokens.total_tokens

            timeline.append(TimelineEntry(
                time_seconds=frame["timestamp"],
                timestamp=frame["timestamp_str"],
                observation=resp.content,
                frame_path=frame_paths[i] if i < len(frame_paths) else None,
            ))

    # Extract entities and generate summary using VLM
    timeline_text = "\n".join(f"{t.timestamp}: {t.observation}" for t in timeline)
    entities_resp = _extract_entities(provider, timeline_text)
    total_tokens.prompt_tokens += entities_resp.tokens.prompt_tokens
    total_tokens.completion_tokens += entities_resp.tokens.completion_tokens
    total_tokens.reasoning_tokens += entities_resp.tokens.reasoning_tokens
    total_tokens.total_tokens += entities_resp.tokens.total_tokens

    if question:
        summary = _build_qa_summary(timeline, question)
    else:
        summary_resp = _generate_summary(provider, timeline)
        summary = summary_resp.content
        total_tokens.prompt_tokens += summary_resp.tokens.prompt_tokens
        total_tokens.completion_tokens += summary_resp.tokens.completion_tokens
        total_tokens.reasoning_tokens += summary_resp.tokens.reasoning_tokens
        total_tokens.total_tokens += summary_resp.tokens.total_tokens

    provider.close()

    # Transcribe audio if requested
    transcript = None
    if transcribe_audio:
        console.print(f"[dim]Transcribing audio with Whisper ({whisper_device})...[/dim]")
        whisper = WhisperProvider({"device": whisper_device})
        import time as _time
        transcript = whisper.transcribe(str(path))

        if transcript and transcript.segments:
            console.print(f"[dim]Transcribed {len(transcript.segments)} speech segments[/dim]")

    # Apply diarization if requested
    if diarize_audio and transcript and transcript.segments:
        try:
            from openvision.providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
            from openvision.core.audio import extract_audio

            console.print("[dim]Running speaker diarization...[/dim]")
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                extract_audio(str(path), output_path=tmp_path)
                diarizer = DiarizerProvider()
                dia_result = diarizer.diarize(tmp_path)
                transcript.segments = merge_transcript_with_diarization(
                    transcript.segments, dia_result.segments
                )
                console.print(f"[dim]Found {dia_result.num_speakers} speakers: {', '.join(dia_result.speakers)}[/dim]")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except ImportError as e:
            console.print(f"[yellow]Diarization unavailable: {e}[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Diarization failed: {e}[/yellow]")

    # Merge transcript segments into timeline (interleaved by timestamp)
    if transcript and transcript.segments:
        audio_entries = []
        for seg in transcript.segments:
            ts = _fmt_time(seg.start)
            speaker_prefix = f"[{seg.speaker}] " if seg.speaker else ""
            audio_entries.append(TimelineEntry(
                time_seconds=seg.start,
                timestamp=ts,
                observation=f"[SPEECH] {speaker_prefix}{seg.text}",
                frame_path=None,
            ))

        # Interleave visual + audio entries by timestamp
        all_entries = timeline + audio_entries
        all_entries.sort(key=lambda e: e.time_seconds)
        timeline = all_entries

    # Build result
    result = ObserveResult(
        summary=summary,
        timeline=timeline,
        entities=entities_resp.entities,
        artifacts_dir=str(run_dir),
        confidence="medium",
        tokens={"prompt_tokens": total_tokens.prompt_tokens, "completion_tokens": total_tokens.completion_tokens, "reasoning_tokens": total_tokens.reasoning_tokens, "total_tokens": total_tokens.total_tokens},
    )

    # Save artifacts
    cache.save_artifact(run_dir, "observations.json", result.model_dump())
    cache.save_text(run_dir, "summary.md", f"# Observation Summary\n\n{summary}\n")

    # Timeline markdown
    timeline_md = "# Timeline\n\n"
    for entry in timeline:
        timeline_md += f"- **{entry.timestamp}** - {entry.observation}\n"
    cache.save_text(run_dir, "timeline.md", timeline_md)

    # Save transcript if available
    if transcript and transcript.segments:
        transcript_text = "\n".join(
            f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
            for s in transcript.segments
        )
        cache.save_text(run_dir, "transcript.txt", transcript_text)

    # Save memory markdown if requested
    if save_memory:
        _save_memory_md(path, result, config, transcript)

    log_vram("observation_complete")

    # Output
    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        _display_observation(result.model_dump())


def _generate_summary(provider, timeline: list) -> VlmResponse:
    """Use VLM to compress the timeline into a concise summary."""
    if not timeline:
        return VlmResponse(content="No observations recorded.", tokens=TokenUsage())

    # Build timeline text (truncated to fit model context limit)
    timeline_text = "\n".join(
        f"{t.timestamp}: {t.observation}" for t in timeline
    )
    # Truncate to ~700 chars to avoid empty responses from LM Studio
    if len(timeline_text) > 700:
        timeline_text = timeline_text[:700] + "..."

    prompt = (
        "Compress this video timeline into 2-3 concise sentences describing "
        "what happened. Focus on key events, objects, and changes.\n\n"
        f"{timeline_text}"
    )

    try:
        resp = provider.query_text(prompt)
        if resp.content and not resp.content.startswith("[VLM"):
            return resp
        return _build_timeline_summary_fallback(timeline)
    except Exception:
        return _build_timeline_summary_fallback(timeline)


def _build_timeline_summary_fallback(timeline: list) -> VlmResponse:
    """Fallback summary if VLM summarization fails."""
    observations = [t.observation for t in timeline if t.observation]
    if not observations:
        return VlmResponse(content="No observations recorded.", tokens=TokenUsage())
    if len(observations) == 1:
        return VlmResponse(content=observations[0], tokens=TokenUsage())
    return VlmResponse(
        content=f"The video shows: {observations[0]} Towards the end, {observations[-1]}",
        tokens=TokenUsage(),
    )


def _extract_entities(provider, timeline_text: str) -> _EntityResult:
    """Use VLM to extract real entities from timeline descriptions."""
    # Truncate to ~700 chars to fit model context limit
    if len(timeline_text) > 700:
        timeline_text = timeline_text[:700] + "..."

    prompt = (
        "From these video observations, extract a comma-separated list of the "
        "main objects, people, and entities visible. Return ONLY the comma-separated list, no explanation.\n\n"
        f"{timeline_text}"
    )
    try:
        resp = provider.query_text(prompt)
        if resp.content and not resp.content.startswith("[VLM"):
            entities = [e.strip().lower() for e in resp.content.split(",") if e.strip()]
            return _EntityResult(entities=entities[:20], tokens=resp.tokens)
        return _EntityResult(entities=[], tokens=TokenUsage())
    except Exception:
        return _EntityResult(entities=[], tokens=TokenUsage())


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
        table = Table()
        table.add_column("Time", style="cyan", width=8)
        table.add_column("Observation", style="white")

        for entry in data["timeline"]:
            obs = entry.get("observation", "")
            ts = entry.get("timestamp", "??:??")
            if len(obs) > 80:
                obs = obs[:80] + "..."
            table.add_row(f"[{ts}]", obs)

        # Only show first 15 rows to avoid flooding
        console.print(table)
        if len(data["timeline"]) > 15:
            console.print(f"  [dim]... and {len(data['timeline']) - 15} more entries[/dim]")

    if data.get("entities"):
        ents = data.get("entities", [])
        console.print(f"\n[bold]Entities:[/bold] {', '.join(ents[:15])}")
        if len(ents) > 15:
            console.print(f"  [dim]... and {len(ents) - 15} more[/dim]")

    if data.get("tokens"):
        tokens = data["tokens"]
        console.print(f"\n[dim]Tokens: {_fmt_tokens(TokenUsage(**tokens))}[/dim]")

    if data.get("artifacts_dir"):
        console.print(f"\n[dim]Artifacts: {data['artifacts_dir']}[/dim]")


def _save_memory_md(path: Path, result: ObserveResult, config: dict,
                    transcript: Optional[Transcript] = None):
    """Save observation as markdown memory file under openvision_HOME (stable path)."""
    from datetime import date
    from openvision.storage.paths import memory_dir as get_memory_dir

    today = date.today().isoformat()
    video_name = path.stem

    mem_dir = get_memory_dir(config)
    memory_path = mem_dir / f"{today}-{video_name}.md"
    lines = [
        f"# Video Observation: {path.name}",
        "",
        f"Date: {today}",
        f"Source: `{path}`",
        f"Artifacts: `{result.artifacts_dir}`",
        "",
        "## Summary",
        "",
        result.summary or "",
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

    # Full transcript when available (not truncated to 30 for vault reuse)
    if transcript and transcript.segments:
        lines.extend(["", "## Transcript", ""])
        for seg in transcript.segments:
            ts = _fmt_time(seg.start)
            speaker = f"[{seg.speaker}] " if getattr(seg, "speaker", None) else ""
            lines.append(f"- **{ts}** {speaker}{seg.text}")

    lines.extend(["", "## Artifacts", "", f"- Frames: `{result.artifacts_dir}/frames/`"])
    lines.extend(["", f"Memory path: `{memory_path}`"])

    memory_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Memory saved:[/green] {memory_path}")
