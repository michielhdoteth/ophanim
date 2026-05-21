"""VisionClaw CLI application."""
import typer
from pathlib import Path
from typing import Optional
from visionclaw.cli.commands import probe, observe, ask, segment, track, status

app = typer.Typer(
    name="visionclaw",
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


def load_config() -> dict:
    """Load config from default.yaml."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
