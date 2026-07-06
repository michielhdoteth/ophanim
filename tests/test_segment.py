"""Tests for SAM provider and segment/track commands."""
import pytest
import numpy as np
from pathlib import Path
from typer.testing import CliRunner
from ophanim.cli.app import app
from ophanim.models import SegmentResult, SegmentObject, TrackResult, TrackPosition
from ophanim.providers.sam import SamProvider

runner = CliRunner()


class TestSegmentModels:
    def test_segment_result_creation(self):
        obj = SegmentObject(
            object_id="person_001",
            timestamps=["00:05", "00:10"],
            mask_paths=["/tmp/mask_1.png"],
            bbox=[100, 200, 300, 400],
        )
        result = SegmentResult(
            prompt="person",
            frames_processed=2,
            masks_dir="/tmp/masks",
            objects=[obj],
        )
        assert result.frames_processed == 2
        assert len(result.objects) == 1


class TestTrackModels:
    def test_track_result_creation(self):
        pos = TrackPosition(
            timestamp="00:05",
            time_seconds=5.0,
            bbox=[100, 200, 300, 400],
            confidence=0.85,
        )
        result = TrackResult(
            track_id="car_test",
            positions=[pos],
            summary="Car tracked across 1 frame.",
        )
        assert result.track_id == "car_test"
        assert len(result.positions) == 1


class TestSamProvider:
    def test_init(self):
        config = {"model": "mobile-sam", "load_on_demand": True}
        provider = SamProvider(config)
        assert provider.model == "mobile-sam"
        assert not provider.is_loaded
        provider.unload()  # Cleanup

    def test_segment_no_model(self):
        """Without SAM installed, should fail gracefully."""
        config = {"model": "mobile-sam", "load_on_demand": True}
        provider = SamProvider(config)

        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        try:
            result = provider.segment(img, "object")
            # If it works, great. If not, should return empty list (not crash)
            assert isinstance(result, list)
        except ImportError:
            pass  # Expected if SAM deps not installed
        finally:
            provider.unload()


class TestSegmentCli:
    def test_segment_nonexistent(self):
        result = runner.invoke(app, ["segment", "nonexistent.mp4", "person"])
        assert result.exit_code != 0

    def test_segment_empty_prompt(self):
        result = runner.invoke(app, ["segment", "test.mp4", ""])
        assert result.exit_code != 0


class TestTrackCli:
    def test_track_nonexistent(self):
        result = runner.invoke(app, ["track", "nonexistent.mp4", "car"])
        assert result.exit_code != 0

    def test_track_empty_prompt(self):
        result = runner.invoke(app, ["track", "test.mp4", ""])
        assert result.exit_code != 0
