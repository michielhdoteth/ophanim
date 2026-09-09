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

from core.video import probe, extract_frames, estimate_processing_cost, auto_fps, _downscale, detect_vfr, detect_color_range
from core.image import downscale, encode_base64, save_frame, load_image
from core.sampling import smart_sample
from core.gpu import auto_downgrade_mode, log_vram
from providers.registry import ProviderRegistry
from providers import get_provider, list_providers
from providers.parakeet import Transcript
from providers.base import VlmResponse, TokenUsage
from storage.cache import RunCache
from storage.config import load_config, get_mode_config
from models import (
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
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda"),
    start_time: str = typer.Option(None, "--start", help="Start time for focus range (e.g., 1:30, 45, 0:15:00)"),
    end_time: str = typer.Option(None, "--end", help="End time for focus range (e.g., 2:00, 90, 0:20:00)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    jsonl: bool = typer.Option(False, "--jsonl", help="Streaming JSONL output for agents/pipelines"),
    save_observations: bool = typer.Option(False, "--save-observations", help="Save observation as markdown ledger"),
    transcribe_audio: bool = typer.Option(False, "--transcribe", "-t", help="Transcribe audio speech to text"),
    stt_provider: str = typer.Option("parakeet", "--stt-provider", help=f"STT provider: {', '.join(list_providers())}"),
    keep_audio: bool = typer.Option(False, "--keep-audio", help="Save full soundtrack as audio.m4a"),
    grid: bool = typer.Option(False, "--grid", help="Generate 3x3 contact sheet from keyframes"),
    text_anchors: bool = typer.Option(False, "--text-anchors", help="Force frames at subtitle-cue timestamps"),
    viewer: bool = typer.Option(False, "--viewer", help="Generate local HTML viewer with keyframes + transcript"),
    report: bool = typer.Option(False, "--report", help="Generate keep/drop frame selection report"),
    dnn_model: str = typer.Option(None, "--dnn-model", help="ONNX model path for inline DNN inference during extraction"),
    cookies: str = typer.Option(None, "--cookies", help="Netscape cookie file for authenticated videos"),
    cookies_from_browser: str = typer.Option(None, "--cookies-from-browser", help="Read cookies from browser (chrome, firefox, edge, safari)"),
    diarize_audio: bool = typer.Option(False, "--diarize", help="Add speaker labels via diarization"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocess (ignore cache)"),
    timestamps: str = typer.Option(None, "--timestamps", help="Extract frames at specific timestamps (comma-separated, e.g., '0:30,2:15,5:00')"),
    provider_name: str = typer.Option(None, "--provider", help="VLM provider: auto, lmstudio, ollama, llamacpp, openai, groq, together, vllm, localai"),
    segment: bool = typer.Option(None, "--segment", "-s", help="Run SAM segmentation on extracted frames (respects mode settings)"),
    raw_frames: bool = typer.Option(False, "--raw-frames", help="Skip VLM — return raw frame paths and audio timeline for vision-capable agents"),
    ground: Optional[str] = typer.Option(None, "--ground", "-g", help="Ground a query using LocateAnything-3B and merge results into the timeline"),
):
    """Analyze a video or image and return observations."""
    # Validate file exists or is URL
    from core.download import is_url
    input_path = Path(path)
    is_remote = is_url(path)
    if not is_remote and not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    # Resolve cookies
    cookies_file = cookies
    if cookies_from_browser and not cookies_file:
        try:
            import browser_cookie3
            import tempfile as _tf
            cj = getattr(browser_cookie3, cookies_from_browser.replace("-", "_").lower())()
            cookie_tmp = _tf.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            cookie_tmp.write("# Netscape HTTP Cookie File\n")
            for c in cj:
                secure = "TRUE" if c.secure else "FALSE"
                domain = c.domain if c.domain.startswith(".") else "." + c.domain
                cookie_tmp.write(f"{domain}\tTRUE\t{c.path}\t{secure}\t{int(c.expires or 0)}\t{c.name}\t{c.value}\n")
            cookie_tmp.close()
            cookies_file = cookie_tmp.name
            console.print(f"[dim]Loaded cookies from {cookies_from_browser}[/dim]")
        except ImportError:
            console.print("[yellow]browser-cookie3 not installed. Install with: pip install browser-cookie3[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Failed to read browser cookies: {e}[/yellow]")

    # Download from URL if remote (native yt-dlp into openvision_HOME/downloads)
    if is_remote:
        from core.download import download_video
        from storage.paths import downloads_dir
        dl_dir = downloads_dir(config)
        console.print(f"[dim]Downloading from URL into {dl_dir}...[/dim]")
        try:
            dl_result = download_video(path, output_dir=str(dl_dir), max_height=720, cookies_file=cookies_file)
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
        from core.video import parse_time
        focus_start = parse_time(start_time)
    if end_time:
        from core.video import parse_time
        focus_end = parse_time(end_time)
    has_focus = focus_start is not None or focus_end is not None

    # Parse cue timestamps
    cue_timestamps = None
    if timestamps:
        from core.video import parse_time
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
        _handle_image(input_path, question, prompt, json_output, config, actual_resolution, max_tokens, provider_name, jsonl)
    else:
        _handle_video(input_path, question, prompt, json_output, save_observations, transcribe_audio, diarize_audio, force,
                      config, mode, detail, actual_fps, actual_max_frames, actual_resolution, max_tokens, device,
                      focus_start, focus_end, cue_timestamps, provider_name, segment, raw_frames, ground, stt_provider,
                      keep_audio, grid, text_anchors, viewer, report, dnn_model, jsonl)


def _handle_image(path: Path, question: Optional[str], custom_prompt: Optional[str],
                  json_output: bool, config: dict, resolution: int,
                  override_max_tokens: Optional[int] = None,
                  provider_name: Optional[str] = None,
                  jsonl: bool = False):
    """Process a single image."""
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_start(str(path))
        writer.emit_probe({"type": "image", "path": str(path)})
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
                  json_output: bool, save_observations: bool, transcribe_audio: bool, diarize_audio: bool, force: bool,
                  config: dict, mode: str, detail: str = "balanced",
                  fps: float = 0.5, max_frames: int = 60, resolution: int = 768,
                  override_max_tokens: Optional[int] = None,
                   stt_device: str = "auto",
                  focus_start: Optional[float] = None,
                  focus_end: Optional[float] = None,
                  cue_timestamps: Optional[list[float]] = None,
                  provider_name: Optional[str] = None,
                  segment_flag: Optional[bool] = None,
                  raw_frames: bool = False,
                  ground_query: Optional[str] = None,
                  stt_provider: str = "parakeet",
                  keep_audio: bool = False,
                  generate_grid: bool = False,
                  text_anchors: bool = False,
                  generate_viewer: bool = False,
                  generate_report: bool = False,
                  dnn_model: str = None,
                  jsonl: bool = False):
    """Process a video file."""
    from core.stream import create_jsonl_writer
    writer = create_jsonl_writer(jsonl)
    if writer:
        writer.emit_start(str(path), mode=mode)

    # Check cache (stable under openvision_HOME when relative)
    cache_dir = config.get("cache", {}).get("directory", "runs")
    if not Path(cache_dir).is_absolute():
        from storage.paths import runs_dir
        cache_dir = str(runs_dir(config))
    cache = RunCache(cache_dir)
    key = cache.cache_key(str(path), mode, fps, resolution)
    mode_config = get_mode_config(config, mode)

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
    if writer:
        writer.emit_probe(video_meta)

    # Detect VFR and color range for smarter sampling
    vfr_info = detect_vfr(str(path))
    vfr_mode = vfr_info.get("mode", "cfr")
    if vfr_mode == "vfr":
        console.print(f"[yellow]VFR detected: {vfr_info.get('variable_frames', 0)} variable / "
                      f"{vfr_info.get('constant_frames', 0)} constant frames "
                      f"({vfr_info.get('vfr_ratio', 0):.0%} VFR)[/yellow]")
        console.print("[dim]Using ffmpeg timestamp-based extraction for accurate sampling.[/dim]")

    color_info = detect_color_range(str(path))
    if color_info.get("range_type") != "unknown":
        console.print(f"[dim]Color range: {color_info['range_type']}[/dim]")

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
        from core.video import auto_fps_focus
        focus_duration = (focus_end or duration) - (focus_start or 0)
        focus_max = auto_fps_focus(focus_duration)
        console.print(f"[dim]Focus range: {_fmt_time(focus_start or 0)} - {_fmt_time(focus_end or duration)} ({focus_duration:.0f}s, budget: {focus_max} frames)[/dim]")
        frames = _extract_focus_range(str(path), focus_start, focus_end, focus_max, resolution)
    elif detail == "transcript":
        console.print("[dim]Transcript-only mode: skipping frame extraction[/dim]")
        frames = []
    elif detail == "efficient":
        from core.video import extract_keyframes_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Efficient mode: extracting keyframes (cap 50)[/dim]")
        frames = extract_keyframes_ffmpeg(str(path), max_frames=50, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    elif detail == "token-burner":
        from core.video import extract_scene_frames_ffmpeg, dedupe_frames_ffmpeg
        console.print("[dim]Token-burner mode: full scene detection (uncapped)[/dim]")
        frames = extract_scene_frames_ffmpeg(str(path), max_frames=9999, max_resolution=resolution)
        frames = dedupe_frames_ffmpeg(frames)
    else:
        # balanced (default)
        if duration > 120:
            from core.sampling import adaptive_sample
            frames = adaptive_sample(str(path), max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)
        else:
            frames = smart_sample(str(path), fps=fps, max_frames=actual_max, max_resolution=resolution, vfr_mode=vfr_mode)

    # Run inline DNN inference if --dnn-model specified
    if dnn_model and frames:
        try:
            from core.dnn_filter import DNNFilterPipeline
            dnn = DNNFilterPipeline(dnn_model)
            if dnn.is_available:
                console.print(f"[dim]Running inline DNN detection: {dnn_model}[/dim]")
                dnn_results = dnn.extract_with_detection(str(path), max_frames=len(frames))
                # Merge detection data into frames
                for dnn_frame in dnn_results:
                    idx = dnn_frame["frame_index"]
                    if idx < len(frames):
                        frames[idx]["detections"] = dnn_frame.get("detections", [])
                        frames[idx]["frame_path_dnn"] = dnn_frame.get("frame_path")
                console.print(f"[dim]DNN: processed {len(dnn_results)} frames[/dim]")
            else:
                console.print("[yellow]FFmpeg DNN/ONNX Runtime not available, skipping inline inference[/yellow]")
        except Exception as e:
            console.print(f"[yellow]DNN filter failed: {e}[/yellow]")

    # Transcript-only early return: skip VLM, just run audio processing
    if detail == "transcript" and not frames:
        run_dir = cache.create_run(key, video_meta)
        transcript = None
        timeline = []

        if transcribe_audio:
            console.print(f"[dim]Transcribing audio with {stt_provider.title()} ({stt_device})...[/dim]")
            whisper = get_provider(stt_provider, {"device": stt_device})
            transcript = whisper.transcribe(str(path))
            if transcript and transcript.segments:
                console.print(f"[dim]Transcribed {len(transcript.segments)} speech segments[/dim]")

        if diarize_audio and transcript and transcript.segments:
            try:
                from providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
                from core.audio import extract_audio

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

        if save_observations:
            _save_observation_md(path, result, config, transcript)

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

    # Generate contact sheet if requested
    if generate_grid and frames:
        try:
            from core.contact_sheet import create_contact_sheet
            grid_path = str(path.parent / f"{path.stem}_contact_sheet.jpg")
            create_contact_sheet(frames, grid_size=(3, 3), output_path=grid_path)
            console.print(f"[dim]Contact sheet: {grid_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Contact sheet failed: {e}[/yellow]")

    # Extract pinned frames at specific timestamps
    pinned_frames = []
    if cue_timestamps:
        from core.video import extract_at_timestamps
        pinned_frames = extract_at_timestamps(str(path), cue_timestamps, resolution)

    # Merge pinned frames into main frame list (dedup by timestamp)
    if pinned_frames:
        existing_times = {round(f["timestamp"], 1) for f in frames}
        for pf in pinned_frames:
            if round(pf["timestamp"], 1) not in existing_times:
                frames.append(pf)
                existing_times.add(round(pf["timestamp"], 1))
        frames.sort(key=lambda f: f["timestamp"])

    # Text-anchors: force frames at subtitle-cue timestamps
    if text_anchors:
        try:
            from core.video import extract_at_timestamps
            # Try to get subtitle timestamps from downloaded subs or captions
            anchor_times = []
            # Check for .srt or .vtt files next to the video
            for ext in [".srt", ".vtt", ".ass"]:
                sub_file = path.with_suffix(ext)
                if sub_file.exists():
                    from core.captions import parse_srt, parse_vtt
                    content = sub_file.read_text(encoding="utf-8", errors="replace")
                    if ext == ".srt":
                        caps = parse_srt(content)
                    else:
                        caps = parse_vtt(content)
                    # Use start time of each cue as an anchor
                    anchor_times = [c.start for c in caps if c.text.strip()]
                    break
            if anchor_times:
                # Cap to avoid too many frames
                if len(anchor_times) > 30:
                    step = len(anchor_times) / 30
                    anchor_times = [anchor_times[int(i * step)] for i in range(30)]
                anchor_frames = extract_at_timestamps(str(path), anchor_times, resolution)
                existing_times = {round(f["timestamp"], 1) for f in frames}
                added = 0
                for af in anchor_frames:
                    if round(af["timestamp"], 1) not in existing_times:
                        frames.append(af)
                        existing_times.add(round(af["timestamp"], 1))
                        added += 1
                if added:
                    frames.sort(key=lambda f: f["timestamp"])
                    console.print(f"[dim]Text-anchors: added {added} frames at subtitle cues[/dim]")
        except Exception as e:
            console.print(f"[yellow]Text-anchors failed: {e}[/yellow]")

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
        if writer:
            writer.emit_frame(i, frame.get("timestamp", 0), str(fpath), frame.get("detections"))

    # Generate keep/drop report if requested
    if generate_report and frames:
        try:
            from core.report import generate_keep_drop_report
            report_path = str(run_dir / "report.html")
            generate_keep_drop_report(
                frames, list(range(len(frames))),
                dedup_stats={"total_extracted": len(frames), "used_for_analysis": len(frames)},
                output_path=report_path,
                title=f"Frame Report - {path.stem}",
            )
            console.print(f"[dim]Report: {report_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Report failed: {e}[/yellow]")

    # --- raw-frames mode: skip VLM, return frames + audio timeline ---
    transcript = None
    if raw_frames:
        from models import RawFrame, TimelineEntry
        raw = [
            RawFrame(index=i, timestamp=f["timestamp"], path=f["path"])
            for i, f in enumerate(frames)
        ]

        # Build audio-only timeline from transcript if available
        timeline: list[TimelineEntry] = []
        if transcript and transcript.segments:
            for seg in transcript.segments:
                timeline.append(TimelineEntry(
                    time_seconds=seg.start,
                    timestamp=_fmt_time(seg.start),
                    observation=seg.text,
                    speaker=getattr(seg, "speaker", None),
                    modality="audio",
                ))

        result = ObserveResult(
            summary=f"Raw frames extracted: {len(frames)} frames from {path.name}",
            timeline=timeline,
            entities=[],
            artifacts_dir=str(run_dir),
            confidence="high",
            raw_frames=raw,
        )

        cache.save_artifact(run_dir, "observations.json", result.model_dump())

        if save_observations:
            _save_observation_md(path, result, config, transcript)

        if json_output:
            console.print(json.dumps(result.model_dump(), indent=2))
        else:
            console.print(f"[green]Extracted {len(frames)} raw frames to {run_dir}[/green]")
            if transcript and transcript.segments:
                console.print(f"[dim]Audio timeline: {len(transcript.segments)} segments with speaker labels[/dim]")
            console.print("[dim]Ready for vision-capable agent processing.[/dim]")
        return

    from models import TimelineEntry

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
        console.print(f"[dim]Transcribing audio with {stt_provider.title()} ({stt_device})...[/dim]")
        whisper = get_provider(stt_provider, {"device": stt_device})
        import time as _time
        transcript = whisper.transcribe(str(path))

        if transcript and transcript.segments:
            console.print(f"[dim]Transcribed {len(transcript.segments)} speech segments[/dim]")

    # Apply diarization if requested
    if diarize_audio and transcript and transcript.segments:
        try:
            from providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
            from core.audio import extract_audio

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

    # Run SAM segmentation if requested or mode requires it
    segmentation_data = None
    seg_mode = mode_config.get("segmentation", False)
    should_segment = segment_flag is True or (segment_flag is None and seg_mode is True)
    if should_segment and frames:
        segmentation_data = _run_segmentation(path, frames, config, mode)

    # Build result
    result = ObserveResult(
        summary=summary,
        timeline=timeline,
        entities=entities_resp.entities,
        artifacts_dir=str(run_dir),
        confidence="medium",
        tokens={"prompt_tokens": total_tokens.prompt_tokens, "completion_tokens": total_tokens.completion_tokens, "reasoning_tokens": total_tokens.reasoning_tokens, "total_tokens": total_tokens.total_tokens},
    )

    # Run LocateAnything grounding if requested
    if ground_query:
        _run_grounding(ground_query, frames, timeline, run_dir, config)

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
        if writer:
            for seg in transcript.segments:
                writer.emit_transcript(seg.start, seg.end, seg.text, getattr(seg, "speaker", None))

    # Save full audio if requested
    if keep_audio:
        try:
            from core.audio import save_full_audio
            audio_out = str(run_dir / "audio.m4a")
            save_full_audio(str(path), audio_out)
            console.print(f"[dim]Audio saved: {audio_out}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Failed to save audio: {e}[/yellow]")

    # Generate HTML viewer if requested
    if generate_viewer and frames:
        try:
            from core.viewer import generate_viewer
            viewer_path = str(run_dir / "viewer.html")
            t_segments = transcript.segments if transcript and transcript.segments else None
            generate_viewer(frames, t_segments, summary, viewer_path, title=path.stem)
            console.print(f"[dim]Viewer: {viewer_path}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Viewer failed: {e}[/yellow]")

    # Save observation markdown if requested
    if save_observations:
        _save_observation_md(path, result, config, transcript)

    # Emit JSONL summary and done
    if writer:
        writer.emit_summary(result.summary, [e.entity for e in result.entities])
        writer.emit_done({"timeline_entries": len(result.timeline), "entities": len(result.entities),
                          "has_transcript": transcript is not None and transcript.transcript is not None})
        writer.flush()

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


def _run_segmentation(path: Path, frames: list, config: dict, mode: str) -> Optional[dict]:
    """Run SAM segmentation on extracted frames. Returns segmentation data or None."""
    try:
        from providers.sam import SamProvider
    except ImportError:
        console.print("[yellow]Segmentation unavailable: install ultralytics (pip install ultralytics)[/yellow]")
        return None

    mode_config = get_mode_config(config, mode)
    resolution = mode_config.get("resolution", 768)
    seg_fps = config.get("defaults", {}).get("segmentation_fps", 0.25)

    log_vram("before_segmentation")
    sam_config = config.get("models", {}).get("segmentation", {})
    provider = SamProvider(sam_config)

    try:
        # Use the entities from VLM to know what to segment, or segment generically
        # For now, segment all frames to detect objects
        seg_frames = [(f["timestamp"], f["image"]) for f in frames[:20]]  # Cap at 20 frames for SAM
        console.print(f"[dim]Running SAM segmentation on {len(seg_frames)} frames...[/dim]")

        # Create temp dir for masks
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path as _P
            result = provider.segment_frames(
                seg_frames,
                "all objects",  # Generic prompt to segment everything
                _P(tmpdir),
            )
            log_vram("after_segmentation")
            return result
    except Exception as e:
        console.print(f"[yellow]Segmentation failed: {e}[/yellow]")
        log_vram("after_segmentation_error")
        return None
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            try:
                provider.unload()
            except Exception:
                pass


def _run_grounding(query: str, frames: list, timeline: list, run_dir, config: dict) -> None:
    """Run LocateAnything grounding and merge results into the timeline."""
    from providers.locate_anything import LocateAnythingProvider
    from models import TimelineEntry

    locate_config = config.get("models", {}).get("locate", {})

    console.print(f"[dim]Running LocateAnything grounding: '{query}'...[/dim]")

    provider = LocateAnythingProvider(locate_config)

    try:
        if not provider.check_health():
            console.print("[yellow]LocateAnything endpoint not available — skipping grounding.[/yellow]")
            return

        log_vram("before_grounding")
        ground_frames = [(f["timestamp"], f["image"]) for f in frames]
        raw_result = provider.locate_frames(ground_frames, query, str(run_dir))
        log_vram("after_grounding")

        # Merge grounding results into timeline
        grounding_timeline = []
        for r in raw_result["results"]:
            if not r["boxes"]:
                continue

            labels = [b["label"] for b in r["boxes"]]
            scores = [f"{b['score']:.0%}" for b in r["boxes"]]

            # Add a grounding entry to the main timeline
            entry = TimelineEntry(
                time_seconds=r["timestamp"],
                timestamp=_fmt_time(r["timestamp"]),
                observation=f"[GROUND] {query}: {', '.join(labels)} (confidence: {', '.join(scores)})",
                frame_path=None,
            )
            timeline.append(entry)
            grounding_timeline.append(entry)

        # Sort timeline by timestamp
        timeline.sort(key=lambda e: e.time_seconds)

        if grounding_timeline:
            console.print(f"[dim]Grounding found {len(grounding_timeline)} frames with matches.[/dim]")
        else:
            console.print(f"[dim]No matches found for '{query}'.[/dim]")

    except Exception as e:
        console.print(f"[yellow]Grounding failed: {e}[/yellow]")
    finally:
        if config.get("gpu_policy", {}).get("unload_after_job", True):
            try:
                provider.unload()
            except Exception:
                pass


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


def _save_observation_md(path: Path, result: ObserveResult, config: dict,
                    transcript: Optional[Transcript] = None):
    """Save observation as markdown ledger file under openvision_HOME (stable path)."""
    from datetime import date
    from storage.paths import observations_dir

    today = date.today().isoformat()
    video_name = path.stem

    obs_dir = observations_dir(config)
    obs_path = obs_dir / f"{today}-{video_name}.md"
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
    lines.extend(["", f"Observation path: `{obs_path}`"])

    obs_path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]Memory saved:[/green] {obs_path}")
