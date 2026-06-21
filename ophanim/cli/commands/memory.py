"""ophanim memory - Manage saved video observations."""
import typer
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from datetime import datetime

console = Console()
MEMORY_DIR = Path("memory") / "videos"


def memory_cmd(
    action: str = typer.Argument("list", help="Action: list, view, delete"),
    name: str = typer.Option(None, "--name", "-n", help="Memory name to view/delete"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Manage saved video observations."""
    if action == "list":
        _list_memories(json_output)
    elif action == "view":
        if not name:
            console.print("[red]Error:[/red] --name is required for view action")
            raise typer.Exit(code=1)
        _view_memory(name)
    elif action == "delete":
        if not name:
            console.print("[red]Error:[/red] --name is required for delete action")
            raise typer.Exit(code=1)
        _delete_memory(name)
    else:
        console.print(f"[red]Unknown action: {action}[/red] (use: list, view, delete)")
        raise typer.Exit(code=1)


def _list_memories(json_output: bool = False):
    """List all saved observation memories."""
    if not MEMORY_DIR.exists():
        console.print("[dim]No saved memories yet. Use --save-memory with observe.[/dim]")
        return

    memories = sorted(MEMORY_DIR.glob("*.md"), reverse=True)

    if not memories:
        console.print("[dim]No saved memories yet.[/dim]")
        return

    if json_output:
        result = []
        for m in memories:
            result.append({
                "name": m.stem,
                "path": str(m),
                "size": m.stat().st_size,
                "modified": datetime.fromtimestamp(m.stat().st_mtime).isoformat(),
            })
        console.print(json.dumps(result, indent=2))
        return

    table = Table(title=f"Saved Memories ({len(memories)})")
    table.add_column("Name", style="cyan")
    table.add_column("Date", style="yellow")
    table.add_column("Size", style="green")

    for m in memories:
        modified = datetime.fromtimestamp(m.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size = m.stat().st_size
        table.add_row(m.stem, modified, f"{size} B")

    console.print(table)


def _view_memory(name: str):
    """View a specific memory file."""
    memory_file = MEMORY_DIR / f"{name}.md"
    if not memory_file.exists():
        # Try partial match
        matches = list(MEMORY_DIR.glob(f"*{name}*.md"))
        if not matches:
            console.print(f"[red]Memory not found: {name}[/red]")
            raise typer.Exit(code=1)
        memory_file = matches[0]

    content = memory_file.read_text(encoding="utf-8")
    console.print(Markdown(content))


def _delete_memory(name: str):
    """Delete a memory file."""
    memory_file = MEMORY_DIR / f"{name}.md"
    if not memory_file.exists():
        matches = list(MEMORY_DIR.glob(f"*{name}*.md"))
        if not matches:
            console.print(f"[red]Memory not found: {name}[/red]")
            raise typer.Exit(code=1)
        memory_file = matches[0]

    memory_file.unlink()
    console.print(f"[green]Deleted:[/green] {memory_file.name}")
