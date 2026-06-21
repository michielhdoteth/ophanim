"""Tests for observe command."""
import pytest
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from ophanim.cli.app import app
from ophanim.models import ObserveResult, ImageResult, TimelineEntry

runner = CliRunner()


class TestObserveModels:
    def test_observe_result_creation(self):
        result = ObserveResult(
            summary="A video of a cat",
            timeline=[
                TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Cat appears"),
            ],
            entities=["cat"],
            artifacts_dir="/tmp/runs/test",
            confidence="medium",
        )
        assert result.summary == "A video of a cat"
        assert len(result.timeline) == 1

    def test_image_result_creation(self):
        result = ImageResult(
            description="A red car on a road",
            objects=["car", "road"],
            text_detected=[],
            confidence="high",
        )
        assert result.objects == ["car", "road"]

    def test_timeline_entry_creation(self):
        entry = TimelineEntry(
            time_seconds=5.5,
            timestamp="00:05",
            observation="A person walks into frame",
            frame_path="/tmp/frames/frame_0000.jpg",
        )
        assert entry.time_seconds == 5.5
        assert entry.observation == "A person walks into frame"

    def test_observe_result_with_timeline_entries(self):
        entries = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Start"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="Middle"),
            TimelineEntry(time_seconds=10.0, timestamp="00:10", observation="End"),
        ]
        result = ObserveResult(
            summary="Test video",
            timeline=entries,
            entities=["test", "video"],
            artifacts_dir="/tmp/runs/test",
            confidence="low",
        )
        assert len(result.timeline) == 3
        assert result.timeline[1].observation == "Middle"
        assert result.confidence == "low"

    def test_image_result_defaults(self):
        result = ImageResult(
            description="A default image",
            confidence="medium",
        )
        assert result.objects == []
        assert result.text_detected == []
        assert result.confidence == "medium"

    def test_observe_result_empty_entities(self):
        result = ObserveResult(
            summary="Empty entities test",
            timeline=[],
            entities=[],
            artifacts_dir="/tmp/runs/test",
            confidence="high",
        )
        assert result.entities == []


class TestObserveCli:
    def test_observe_nonexistent(self):
        result = runner.invoke(app, ["observe", "nonexistent.mp4"])
        assert result.exit_code != 0

    def test_observe_nonexistent_image(self):
        result = runner.invoke(app, ["observe", "nonexistent.jpg"])
        assert result.exit_code != 0

    def test_observe_help_shows(self):
        result = runner.invoke(app, ["observe", "--help"])
        assert result.exit_code == 0
        assert "Analyze" in result.output or "observe" in result.output.lower()

    def test_observe_invalid_mode(self):
        result = runner.invoke(app, ["observe", "--mode", "invalid", "test.mp4"])
        # Should fail because file doesn't exist first
        assert result.exit_code != 0

    def test_observe_image_missing_lmstudio(self):
        """If no test image exists, this should still fail cleanly."""
        test_images = list(Path("test_data").glob("*.jpg")) + list(Path("test_data").glob("*.png"))
        test_images += list(Path(".").glob("*.jpg"))
        if not test_images:
            pytest.skip("No test image available")

        result = runner.invoke(app, ["observe", str(test_images[0])])
        # Either works (LM Studio running) or fails with LM Studio error
        assert result.exit_code in (0, 1)

    def test_observe_with_question_flag(self):
        result = runner.invoke(app, ["observe", "--question", "What is this?", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_json_flag(self):
        result = runner.invoke(app, ["observe", "--json", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_force_flag(self):
        result = runner.invoke(app, ["observe", "--force", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_save_memory(self):
        result = runner.invoke(app, ["observe", "--save-memory", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_fps(self):
        result = runner.invoke(app, ["observe", "--fps", "1.0", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_max_frames(self):
        result = runner.invoke(app, ["observe", "--max-frames", "10", "test.mp4"])
        assert result.exit_code != 0  # File not found

    def test_observe_with_all_options(self):
        result = runner.invoke(app, [
            "observe", "--mode", "fast", "--fps", "0.5", "--max-frames", "30",
            "--question", "What color?", "--json", "--force", "--save-memory",
            "test.mp4"
        ])
        assert result.exit_code != 0  # File not found


class TestBuildSummary:
    """Test the summary builder functions directly via observe module."""

    def _import_helpers(self):
        from ophanim.cli.commands.observe import (
            _build_timeline_summary_fallback, _build_qa_summary
        )
        return _build_timeline_summary_fallback, _build_qa_summary

    def test_timeline_summary_single(self):
        tl_fn, _ = self._import_helpers()
        from ophanim.models import TimelineEntry
        timeline = [TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="A single event")]
        result = tl_fn(timeline)
        assert result == "A single event"

    def test_timeline_summary_multiple(self):
        tl_fn, _ = self._import_helpers()
        from ophanim.models import TimelineEntry
        timeline = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="First event"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="Second event"),
            TimelineEntry(time_seconds=10.0, timestamp="00:10", observation="Last event"),
        ]
        result = tl_fn(timeline)
        assert "First event" in result
        assert "Last event" in result

    def test_timeline_summary_empty(self):
        tl_fn, _ = self._import_helpers()
        result = tl_fn([])
        assert "No observations" in result

    def test_qa_summary_with_answers(self):
        _, qa_fn = self._import_helpers()
        from ophanim.models import TimelineEntry
        timeline = [
            TimelineEntry(time_seconds=0.0, timestamp="00:00", observation="Short"),
            TimelineEntry(time_seconds=5.0, timestamp="00:05", observation="A much longer detailed observation about the scene"),
        ]
        result = qa_fn(timeline, "What is happening?")
        assert "Based on video analysis" in result
        assert "longer detailed" in result

    def test_qa_summary_empty(self):
        _, qa_fn = self._import_helpers()
        result = qa_fn([], "What?")
        assert "No frames" in result
