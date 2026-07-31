"""openvision status - Show system status."""
import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from openvision.core.gpu import get_vram_info, get_gpu_name, check_safe_mode
from openvision.storage.cache import RunCache
from openvision.storage.config import load_config
from openvision.models import StatusResult
from openvision.providers.registry import ProviderRegistry

console = Console()


def status_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show openvision system status, GPU state, and cache info."""
    try:
        config = load_config()
    except FileNotFoundError:
        config = {}

    # GPU info
    gpu_name = get_gpu_name()
    vram = get_vram_info()
    safety = check_safe_mode(config)

    # Paths + cache (stable under openvision_HOME)
    from openvision.storage.paths import get_home, observations_dir, downloads_dir, runs_dir

    home = get_home(config)
    obs = observations_dir(config)
    dls = downloads_dir(config)
    runs_path = runs_dir(config)

    cache_dir = config.get("cache", {}).get("directory", "runs")
    if not Path(cache_dir).is_absolute():
        cache_dir = str(runs_path)
    run_cache = RunCache(cache_dir)
    runs = run_cache.list_runs()

    result = StatusResult(
        gpu=gpu_name or "unknown",
        vram_total_gb=round(vram["total_gb"], 1) if vram["total_gb"] > 0 else 0.0,
        vram_free_gb=round(vram["free_gb"], 1) if vram["free_gb"] > 0 else 0.0,
        loaded_models=_get_loaded_models(config),
        safe_mode=safety.get("safe_mode", True),
        queue=0,
    )

    if json_output:
        payload = result.model_dump()
        payload["paths"] = {
            "home": str(home),
            "observations": str(obs),
            "downloads": str(dls),
            "runs": str(runs_path),
        }
        # Add provider health to JSON output
        try:
            detected = ProviderRegistry.detect()
            payload["providers"] = detected
        except Exception:
            payload["providers"] = []
        console.print(json.dumps(payload, indent=2))
    else:
        _display_status(result, runs, safety)
        # Provider health panel
        _display_provider_health()
        path_table = Table(show_header=False, box=None, title="Data Paths")
        path_table.add_column("Key", style="cyan")
        path_table.add_column("Path")
        path_table.add_row("Home", str(home))
        path_table.add_row("Observations", str(obs))
        path_table.add_row("Downloads", str(dls))
        path_table.add_row("Runs", str(runs_path))
        console.print(path_table)
        obs_count = len(list(obs.glob("*.md")))
        console.print(f"[dim]Saved observations: {obs_count}[/dim]")


def _get_loaded_models(config: dict) -> list[str]:
    """Get list of configured models."""
    models = []
    vlm = config.get("models", {}).get("vlm", {})
    if vlm.get("provider"):
        models.append(f"{vlm['provider']}:{vlm.get('model', 'unknown')}")
    return models


def _display_status(result: StatusResult, runs: list[dict], safety: dict):
    """Display status in rich format."""
    # GPU Panel
    gpu_table = Table(show_header=False, box=None)
    gpu_table.add_column("Property", style="cyan")
    gpu_table.add_column("Value")

    gpu_table.add_row("GPU", result.gpu)
    gpu_table.add_row("VRAM", f"{result.vram_used_gb:.1f} / {result.vram_total_gb:.1f} GB"
                      if hasattr(result, 'vram_used_gb') else
                      f"{result.vram_free_gb:.1f} GB free")

    free_gb = safety.get("free_gb", 0)
    total_gb = safety.get("total_gb", 0)
    used = total_gb - free_gb
    gpu_table.add_row("VRAM Used", f"{used:.1f} / {total_gb:.1f} GB" if total_gb > 0 else "Unknown")
    gpu_table.add_row("VRAM Free", f"{free_gb:.1f} GB" if free_gb > 0 else "Unknown")

    safe_color = "green" if not result.safe_mode else "red"
    gpu_table.add_row("Safe Mode", f"[{safe_color}]{'ON' if result.safe_mode else 'OFF'}[/{safe_color}]")
    gpu_table.add_row("Safety", safety.get("reason", "Unknown"))

    if result.loaded_models:
        gpu_table.add_row("Configured Models", ", ".join(result.loaded_models))

    console.print(Panel(gpu_table, title="[bold cyan]System Status[/bold cyan]"))

    # Cache Panel
    if runs:
        cache_table = Table(title=f"Cached Runs ({len(runs)})")
        cache_table.add_column("Run", style="cyan")
        cache_table.add_column("Date")
        cache_table.add_column("Path")

        for run in runs[:10]:  # Show last 10
            from datetime import datetime
            created = datetime.fromtimestamp(run["created"]).strftime("%Y-%m-%d %H:%M")
            cache_table.add_row(run["name"][:30], created, run["path"])

        console.print(cache_table)
    else:
        console.print("[dim]No cached runs yet.[/dim]")


def _display_provider_health():
    """Display health status of all VLM providers."""
    try:
        detected = ProviderRegistry.detect()
    except Exception as e:
        console.print(f"[dim]Could not probe providers: {e}[/dim]")
        return

    if not detected:
        console.print("[dim]No VLM providers detected.[/dim]")
        return

    provider_table = Table(title="VLM Providers")
    provider_table.add_column("Provider", style="cyan")
    provider_table.add_column("Status")
    provider_table.add_column("URL")
    provider_table.add_column("Models")

    for info in detected:
        name = info.get("name", "unknown")
        status = info.get("status", "unknown")
        url = info.get("url", "")
        models = info.get("models", [])
        error = info.get("error", "")

        if status == "available":
            status_display = "[green]Available[/green]"
            models_display = ", ".join(models[:3]) if models else "none"
        elif status == "timeout":
            status_display = "[yellow]Timeout[/yellow]"
            models_display = error
        else:
            status_display = "[red]Unavailable[/red]"
            models_display = error

        provider_table.add_row(name, status_display, url, models_display)

    console.print(provider_table)
