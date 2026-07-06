"""Ophanim CLI application."""
import typer
import sys
from pathlib import Path
from typing import Optional
from ophanim.cli.commands import probe, observe, ask, segment, track, status, memory, transcribe
from ophanim.core.errors import OphanimError, handle_cli_error

app = typer.Typer(
    name="ophanim",
    help="Local visual perception layer for agents",
    no_args_is_help=True,
)

# Register commands
app.command(name="probe")(probe.probe_cmd)
app.command(name="observe")(observe.observe_cmd)
app.command(name="ask")(ask.ask_cmd)
app.command(name="segment")(segment.segment_cmd)
app.command(name="track")(track.track_cmd)
app.command(name="status")(status.status_cmd)
app.command(name="memory")(memory.memory_cmd)
app.command(name="transcribe")(transcribe.transcribe_cmd)


def load_config() -> dict:
    """Load config from default.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Entry point for the CLI."""
    try:
        app()
    except OphanimError as e:
        error_info = e.to_dict()
        if "--json" in sys.argv:
            import json
            from rich.console import Console
            console = Console()
            console.print(json.dumps(error_info, indent=2))
        else:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            console.print(Panel(
                f"[red]{e.code}[/red]\n{e.message}",
                title="Error",
                border_style="red",
            ))
            if e.suggested_retry:
                console.print("[yellow]Suggested retry:[/yellow]")
                import json as _json
                console.print(_json.dumps(e.suggested_retry, indent=2))
        sys.exit(1)
    except Exception as e:
        from rich.console import Console
        console = Console()
        error_info = handle_cli_error(e)
        console.print(f"[red]Error:[/red] {error_info.get('message', str(e))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
