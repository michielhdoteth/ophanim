"""ophanim probe <path> - Extract video metadata."""
import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from ophanim.core.video import probe, estimate_processing_cost
from ophanim.models import ProbeResult

console = Console()


def probe_cmd(
    path: str = typer.Argument(..., help="Path to video file", exists=True),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Extract metadata from a video file without processing it.
    
    Returns duration, resolution, FPS, codec, and estimated processing cost.
    """
    # Validate file exists
    video_path = Path(path)
    if not video_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)
    
    # Probe the video
    try:
        metadata = probe(str(video_path))
    except Exception as e:
        console.print(f"[red]Error probing video:[/red] {e}")
        raise typer.Exit(code=1)
    
    # Estimate processing cost
    cost = estimate_processing_cost(metadata)
    metadata["estimated_processing_cost"] = cost
    
    result = ProbeResult(**metadata)
    
    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        _display_probe_table(result, str(video_path))


def _display_probe_table(result: ProbeResult, path: str):
    """Display probe results in a rich table."""
    table = Table(title=f"Video Probe: {path}", title_style="bold cyan")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Duration", f"{result.duration_seconds:.1f}s ({_format_duration(result.duration_seconds)})")
    table.add_row("Resolution", f"{result.width}x{result.height}")
    table.add_row("FPS", f"{result.fps:.2f}")
    table.add_row("Codec", result.codec)
    table.add_row("Frame Count", str(result.frame_count))
    table.add_row("Estimated Cost", result.estimated_processing_cost.upper())
    
    console.print(table)
    
    # Color-coded cost warning
    cost_colors = {"low": "green", "medium": "yellow", "high": "red"}
    color = cost_colors.get(result.estimated_processing_cost, "white")
    console.print(f"\nEstimated processing cost: [{color}]{result.estimated_processing_cost.upper()}[/{color}]")


def _format_duration(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"
