"""ophanim ask <path> <question> - Ask a specific question about a video."""
import typer
import json
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from ophanim.core.video import probe
from ophanim.core.sampling import smart_sample, find_keyframes_for_question
from ophanim.providers.lmstudio import LmStudioProvider
from ophanim.storage.config import load_config, get_mode_config
from ophanim.models import AskResult

console = Console()


def ask_cmd(
    path: str = typer.Argument(..., help="Path to video file"),
    question: str = typer.Argument(..., help="Question to answer about the video"),
    mode: str = typer.Option("balanced", "--mode", "-m", help="Search mode: keyframes, scene_changes, dense"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Ask a targeted question about video content."""
    input_path = Path(path)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(code=1)

    if not question.strip():
        console.print("[red]Error:[/red] Question cannot be empty.")
        raise typer.Exit(code=1)

    # Load config
    config = load_config()
    mode_config = get_mode_config(config, mode)

    fps = mode_config.get("fps", 0.5)
    max_frames = mode_config.get("max_frames", 60)
    resolution = mode_config.get("resolution", 768)

    # Determine search mode
    search_mode = None
    if mode == "dense":
        search_mode = "dense"
    elif mode == "scene_changes":
        search_mode = "scenes"
    else:
        search_mode = "keyframes"  # default

    # Adjust sampling based on search mode
    if search_mode == "dense":
        dense_fps = min(fps * 3, 2.0)  # Max 2 FPS for dense
        frames = smart_sample(str(input_path), fps=dense_fps, max_frames=max_frames * 2, max_resolution=resolution)
    elif search_mode == "scenes":
        frames = smart_sample(str(input_path), fps=fps, max_frames=max_frames, max_resolution=resolution, scene_threshold=15.0)  # More sensitive
    else:
        frames = smart_sample(str(input_path), fps=fps, max_frames=max_frames, max_resolution=resolution)

    if not frames:
        console.print("[red]No frames extracted from video.[/red]")
        raise typer.Exit(code=1)

    console.print(f"[dim]Analyzing {len(frames)} frames to answer: {question}[/dim]")

    # Query VLM
    vlm_config = config.get("models", {}).get("vlm", {})
    provider = LmStudioProvider(vlm_config)

    if not provider.check_health():
        console.print("[red]Error: LM Studio is not running.[/red]")
        console.print("Start LM Studio and load google/gemma-4-e2b, then try again.")
        raise typer.Exit(code=1)

    evidence = []

    for frame in frames:
        try:
            answer = provider.describe_image(frame["image"], question)
        except Exception as e:
            console.print(f"[dim]Error on frame {frame['timestamp_str']}: {e}[/dim]")
            continue

        # Only include frames that provide useful answers (not empty, not "I don't know")
        if answer and len(answer) > 5 and not _is_uncertain(answer):
            evidence.append({
                "timestamp": frame["timestamp_str"],
                "time_seconds": frame["timestamp"],
                "answer": answer,
                "confidence": "medium",
            })

        # Stop early if we found strong evidence
        if len(evidence) >= 3:
            console.print(f"[dim]Found sufficient evidence, stopping early.[/dim]")
            break

    provider.close()

    # Compile final answer from evidence
    if evidence:
        # Use the most detailed answer as primary
        best = max(evidence, key=lambda e: len(e["answer"]))
        final_answer = best["answer"]
        confidence = "high" if len(evidence) >= 2 else "medium"

        # Add synthesis if multiple evidence points
        if len(evidence) > 1:
            timestamps = [e["timestamp"] for e in evidence]
            final_answer += f" (observed at {', '.join(timestamps)})"
    else:
        final_answer = f"Could not find evidence in the video to answer: {question}"
        confidence = "low"

    result = AskResult(
        answer=final_answer,
        evidence=evidence,
        confidence=confidence,
    )

    if json_output:
        console.print(json.dumps(result.model_dump(), indent=2))
    else:
        _display_ask_result(result, question)


def _is_uncertain(answer: str) -> bool:
    """Check if a VLM response indicates uncertainty."""
    uncertain_phrases = [
        "i cannot", "i can't", "unable to", "no information",
        "not visible", "cannot determine", "not clear",
        "i don't know", "there is no", "does not appear",
        "insufficient", "unclear", "not enough",
    ]
    lower = answer.lower()
    return any(phrase in lower for phrase in uncertain_phrases)


def _display_ask_result(result: AskResult, question: str):
    """Display ask result in rich format."""
    # Color by confidence
    colors = {"high": "green", "medium": "yellow", "low": "red"}
    color = colors.get(result.confidence, "white")

    console.print(f"\n[bold]Question:[/bold] {question}")
    console.print(f"[bold]Confidence:[/bold] [{color}]{result.confidence.upper()}[/{color}]")
    console.print(Panel(result.answer, title="[bold cyan]Answer[/bold cyan]"))

    if result.evidence:
        console.print("\n[bold]Evidence:[/bold]")
        for ev in result.evidence:
            ts = ev.get("timestamp", "??:??")
            ans = ev.get("answer", "")
            if len(ans) > 120:
                ans = ans[:120] + "..."
            console.print(f"  [dim][{ts}][/dim] {ans}")
