"""Install and manage STT models for Open Vision."""
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core.models import (
    list_all_models,
    check_model_status,
    install_model,
    models_dir,
)

console = Console()
app = typer.Typer(help="Install and manage STT models")


def _format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    if size_bytes < 1_000:
        return f"{size_bytes} B"
    elif size_bytes < 1_000_000:
        return f"{size_bytes / 1_000:.1f} KB"
    elif size_bytes < 1_000_000_000:
        return f"{size_bytes / 1_000_000:.1f} MB"
    else:
        return f"{size_bytes / 1_000_000_000:.1f} GB"


@app.command()
def list_models():
    """List all available models and their install status."""
    models = list_all_models()

    table = Table(title="STT Models")
    table.add_column("Model", style="cyan")
    table.add_column("Provider", style="green")
    table.add_column("Size", justify="right")
    table.add_column("Status")
    table.add_column("Description")

    for m in models:
        status = "[green]Installed[/green]" if m.installed else "[red]Not installed[/red]"
        size = _format_size(m.size_bytes)
        table.add_row(m.name, m.provider, size, status, m.description)

    console.print(table)
    console.print(f"\nModels directory: {models_dir()}")


@app.command()
def get(
    provider: str = typer.Argument(
        help="Provider: parakeet or whisper",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Model name (e.g., base, small, large-v3 for whisper)",
    ),
    all_models: bool = typer.Option(
        False, "--all", "-a",
        help="Install all models (parakeet + whisper variants)",
    ),
):
    """Install STT models for Open Vision.

    Examples:

        openvision install parakeet                        # Install parakeet (default)
        openvision install whisper --model base            # Install whisper base
        openvision install parakeet --all                  # Install all models
    """
    if all_models:
        _install_all()
        return

    if provider == "parakeet":
        _install_parakeet()
    elif provider == "whisper":
        if model is None:
            console.print("[yellow]Whisper requires a model size. Use --model base/small/medium/large-v3[/yellow]")
            raise typer.Exit(1)
        _install_whisper(model)
    else:
        console.print(f"[red]Unknown provider: {provider}. Use 'parakeet' or 'whisper'.[/red]")
        raise typer.Exit(1)


def _install_parakeet():
    """Install the Parakeet model."""
    status = check_model_status("parakeet", "parakeet-tdt-0.6b-v3-int8")
    if status.installed:
        console.print(f"[green]Parakeet model already installed at {status.path}[/green]")
        return

    try:
        path = install_model("parakeet", "parakeet-tdt-0.6b-v3-int8")
        console.print(f"[green]Parakeet model installed to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to install Parakeet model: {e}[/red]")
        raise typer.Exit(1)


def _install_whisper(model_size: str):
    """Install a Whisper model."""
    status = check_model_status("whisper", model_size)
    if status.installed:
        console.print(f"[green]Whisper {model_size} already installed at {status.path}[/green]")
        return

    try:
        path = install_model("whisper", model_size)
        console.print(f"[green]Whisper {model_size} installed to {path}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to install Whisper {model_size}: {e}[/red]")
        raise typer.Exit(1)


def _install_all():
    """Install all models."""
    console.print(Panel("Installing all STT models", title="Open Vision Install"))

    console.print("\n[bold]1/6 Parakeet TDT 0.6B v3[/bold]")
    _install_parakeet()

    whisper_sizes = ["tiny", "base", "small", "medium", "large-v3"]
    for i, size in enumerate(whisper_sizes, 2):
        console.print(f"\n[bold]{i}/6 Whisper {size}[/bold]")
        _install_whisper(size)

    console.print("\n[green bold]All models installed.[/green bold]")
