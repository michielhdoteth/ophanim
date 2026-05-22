"""ophanim transcribe <path> - Transcribe audio from video."""
import time
import sys
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ophanim.providers.whisper import WhisperProvider

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
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    save_text: bool = typer.Option(False, "--save", help="Save transcript to text file"),
):
    """Transcribe audio from a video file using Whisper."""
    input_path = Path(path)
    if not input_path.exists():
        _safe_print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    _safe_print(f"[dim]Loading Whisper model '{model}' on CPU...[/dim]")

    config = {
        "model_size": model,
        "device": "cpu",
        "compute_type": "int8",
    }
    provider = WhisperProvider(config)

    try:
        _safe_print("[dim]Transcribing audio... (this may take a while)[/dim]")
        t0 = time.time()
        transcript = provider.transcribe(str(input_path))
        elapsed = time.time() - t0
        _safe_print(f"[dim]Transcription finished in {elapsed:.1f}s[/dim]")
    except Exception as e:
        _safe_print(f"[red]Error during transcription:[/red] {e}")
        raise typer.Exit(code=1)

    if not transcript.segments:
        _safe_print("[yellow]No speech detected in the video audio track.[/yellow]")
        return

    # Build output
    if json_output:
        output = {
            "language": transcript.language,
            "duration_seconds": transcript.duration_seconds,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text, "confidence": s.confidence}
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
        txt_path.write_text(transcript.text, encoding="utf-8")
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
    table.add_column("Text", style="white")
    table.add_column("Conf", style="green", width=6)

    for seg in transcript.segments:
        ts = f"{_fmt_time(seg.start)} -> {_fmt_time(seg.end)}"
        conf = f"{seg.confidence:.2f}" if seg.confidence else "-"
        text = seg.text[:80] + "..." if len(seg.text) > 80 else seg.text
        table.add_row(ts, text, conf)

    console.print(table)


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
