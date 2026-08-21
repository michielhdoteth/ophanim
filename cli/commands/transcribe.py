"""openvision transcribe <path> - Transcribe audio from video."""
import os
import tempfile
import time
import sys
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from providers.whisper import WhisperProvider

console = Console()


def _safe_print(msg: str):
    """Print a message safely, falling back to plain print on Unicode errors."""
    try:
        console.print(msg)
    except UnicodeEncodeError:
        # Strip Rich markup for plain-text fallback
        import re
        plain = re.sub(r'\[/?\w+(?:=.*?)?\]', '', msg)
        print(plain)


def transcribe_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    language: str = typer.Option(None, "--language", "-l", help="Language code (e.g., 'en'). Auto-detect if not set."),
    model: str = typer.Option("base", "--model", "-m", help="Whisper model size: tiny, base, small, medium, large"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cpu, cuda"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    save_text: bool = typer.Option(False, "--save", help="Save transcript to text file"),
    diarize_audio: bool = typer.Option(False, "--diarize", help="Add speaker labels via diarization"),
    min_speakers: int = typer.Option(None, "--min-speakers", help="Minimum number of speakers"),
    max_speakers: int = typer.Option(None, "--max-speakers", help="Maximum number of speakers"),
):
    """Transcribe audio from a video file using Whisper."""
    from core.download import is_url
    input_path = Path(path)
    is_remote = is_url(path)
    if not is_remote and not input_path.exists():
        _safe_print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    _safe_print(f"[dim]Loading Whisper model '{model}' on {device}...[/dim]")

    config = {
        "model_size": model,
        "device": device,
    }
    provider = WhisperProvider(config)

    # Download from URL if remote (native yt-dlp into stable downloads dir)
    dl_result = {}
    if is_remote:
        from core.download import download_video
        from storage.paths import downloads_dir
        from storage.config import load_config
        try:
            cfg = load_config()
        except Exception:
            cfg = None
        dl_dir = downloads_dir(cfg)
        _safe_print(f"[dim]Downloading from URL into {dl_dir}...[/dim]")
        try:
            dl_result = download_video(
                path, output_dir=str(dl_dir), audio_only=False, write_subs=True
            )
            input_path = Path(dl_result["path"])
            _safe_print(f"[dim]Downloaded: {dl_result.get('title', input_path.name)}[/dim]")
        except Exception as e:
            _safe_print(f"[red]Download failed:[/red] {e}")
            raise typer.Exit(code=1)

    # Check for downloaded subtitle files
    if is_remote and dl_result.get("subs_file") and os.path.exists(dl_result["subs_file"]):
        try:
            from core.captions import parse_vtt, filter_range
            sub_content = Path(dl_result["subs_file"]).read_text(encoding="utf-8", errors="replace")
            caps = parse_vtt(sub_content)
            if caps:
                from providers.whisper import Transcript, TranscriptSegment
                transcript = Transcript(
                    segments=[TranscriptSegment(start=c.start, end=c.end, text=c.text, confidence=1.0) for c in caps],
                    language="en",
                    duration_seconds=caps[-1].end if caps else 0,
                )
                _safe_print(f"[green]Using downloaded captions ({len(transcript.segments)} segments)[/green]")
            else:
                transcript = None
        except Exception as e:
            _safe_print(f"[dim]Caption parsing failed: {e}[/dim]")
            transcript = None
    else:
        transcript = None

    # Try existing captions first (faster, no GPU needed)
    if transcript is None:
        try:
            captions = provider.try_captions(str(input_path))
            if captions and captions.segments:
                _safe_print(f"[green]Found embedded captions ({len(captions.segments)} segments)[/green]")
                transcript = captions
            else:
                _safe_print("[dim]No embedded captions found, transcribing with Whisper...[/dim]")
                t0 = time.time()
                transcript = provider.transcribe(str(input_path))
                elapsed = time.time() - t0
                _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")
        except Exception as e:
            _safe_print(f"[dim]Caption extraction failed ({e}), transcribing with Whisper...[/dim]")
            t0 = time.time()
            transcript = provider.transcribe(str(input_path))
            elapsed = time.time() - t0
            _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")

    if not transcript.segments:
        _safe_print("[yellow]No speech detected in the video audio track.[/yellow]")
        return

    # Apply diarization if requested
    if diarize_audio and transcript.segments:
        try:
            from providers.diarizer import DiarizerProvider, merge_transcript_with_diarization
            from core.audio import extract_audio

            _safe_print("[dim]Running speaker diarization...[/dim]")

            # Extract audio to temp WAV for diarizer
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                extract_audio(str(input_path), output_path=tmp_path)
                diarizer = DiarizerProvider()
                dia_result = diarizer.diarize(
                    tmp_path,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                transcript.segments = merge_transcript_with_diarization(
                    transcript.segments, dia_result.segments
                )
                _safe_print(f"[dim]Found {dia_result.num_speakers} speakers: {', '.join(dia_result.speakers)}[/dim]")
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError as e:
            _safe_print(f"[yellow]Diarization unavailable: {e}[/yellow]")
        except Exception as e:
            _safe_print(f"[yellow]Diarization failed: {e}[/yellow]")

    # Build output
    if json_output:
        output = {
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "confidence": s.confidence, "speaker": s.speaker}
                for s in transcript.segments
            ],
            "full_text": transcript.text,
        }
        console.print(json.dumps(output, indent=2))
    else:
        _display_transcript(transcript)

    # Optionally save
    if save_text:
        txt_path = input_path.with_suffix(".txt")
        lines = [
            f"[{_fmt_time(s.start)}] {'[' + s.speaker + '] ' if s.speaker else ''}{s.text}"
            for s in transcript.segments
        ]
        txt_path.write_text("\n".join(lines), encoding="utf-8")
        _safe_print(f"[green]Transcript saved:[/green] {txt_path}")


def _display_transcript(transcript):
    """Display transcript in rich format."""
    # Summary
    summary = (
        f"Language: {transcript.language} | "
        f"Duration: {transcript.duration_seconds:.0f}s | "
        f"Segments: {len(transcript.segments)}"
    )
    console.print(Panel(summary, title="[bold cyan]Transcription[/bold cyan]"))

    # Segments table
    table = Table()
    table.add_column("Time", style="cyan", width=14)
    table.add_column("Speaker", style="magenta", width=12)
    table.add_column("Text", style="white")
    table.add_column("Conf", style="green", width=6)

    for seg in transcript.segments:
        ts = f"{_fmt_time(seg.start)} -> {_fmt_time(seg.end)}"
        speaker = seg.speaker if seg.speaker else "-"
        conf = f"{seg.confidence:.2f}" if seg.confidence else "-"
        text = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
        table.add_row(ts, speaker, text, conf)

    console.print(table)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
