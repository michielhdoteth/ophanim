"""Print the LLM agent skill guide."""
import typer
from pathlib import Path


def skill_cmd():
    """Print the full LLM agent skill guide for openvision.

    Shows all commands, flags, workflows, and backend selection.
    """
    skill_path = Path(__file__).parent.parent.parent / "SKILL.md"
    if skill_path.exists():
        typer.echo(skill_path.read_text())
    else:
        typer.echo("SKILL.md not found in project root", err=True)
        raise typer.Exit(1)
