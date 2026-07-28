"""Tests for core video and image modules."""
import pytest
import numpy as np
from pathlib import Path
from openvision.core.video import (
    probe,
    _format_timestamp,
    _downscale,
    estimate_processing_cost,
    auto_fps_focus,
    extract_at_timestamps,
)
from openvision.core.image import (
    downscale,
    encode_base64,
    make_thumbnail,
    load_image,
)


class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "00:00"

    def test_minutes(self):
        assert _format_timestamp(125) == "02:05"

    def test_hours_not_expected(self):
        # Only MM:SS for now
        result = _format_timestamp(3661)
        assert result == "61:01"  # 61 minutes, 1 second


class TestDownscale:
    def test_no_resize_needed(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape == (100, 200, 3)

    def test_downscales_width(self):
        img = np.zeros((100, 1600, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape[1] <= 768  # width shrunk
        assert result.shape[0] < 100   # height shrunk proportionally

    def test_downscales_height(self):
        img = np.zeros((1600, 100, 3), dtype=np.uint8)
        result = _downscale(img, 768)
        assert result.shape[0] <= 768
        assert result.shape[1] < 100


class TestEncodeBase64:
    def test_encodes_jpg(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = encode_base64(img, ".jpg")
        assert result.startswith("data:image/jpeg;base64,")
        assert len(result) > 50

    def test_encodes_png(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        result = encode_base64(img, ".png")
        assert result.startswith("data:image/png;base64,")


class TestMakeThumbnail:
    def test_thumbnail_smaller(self):
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        thumb = make_thumbnail(img, 256)
        assert max(thumb.shape[:2]) <= 256


class TestEstimateProcessingCost:
    def test_low_cost(self):
        meta = {"duration_seconds": 30, "width": 640, "height": 480}
        assert estimate_processing_cost(meta, "balanced") == "low"

    def test_high_cost_long_video(self):
        meta = {"duration_seconds": 600, "width": 1920, "height": 1080}
        assert estimate_processing_cost(meta, "detailed") == "high"


class TestLoadImage:
    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.jpg")


class TestProbeIntegration:
    """Integration test - needs a real video file."""

    def test_probe_real_video(self):
        """If there's a test video, probe it. Otherwise skip."""
        import os
        test_video = "test_data/sample.mp4"
        if not os.path.exists(test_video) and not os.path.exists("test_data"):
            pytest.skip("No test video available")

        # Use the first .mp4 found in test_data
        test_data = Path("test_data")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        meta = probe(str(videos[0]))
        assert meta["duration_seconds"] > 0
        assert meta["width"] > 0
        assert meta["height"] > 0
        assert meta["fps"] > 0


class TestAutoFpsFocus:
    def test_5s_returns_30(self):
        assert auto_fps_focus(5) == 30

    def test_1s_returns_30(self):
        assert auto_fps_focus(1) == 30

    def test_0_returns_30(self):
        assert auto_fps_focus(0) == 30

    def test_10s_returns_60(self):
        assert auto_fps_focus(10) == 60

    def test_15s_returns_60(self):
        assert auto_fps_focus(15) == 60

    def test_20s_returns_60(self):
        assert auto_fps_focus(20) == 60

    def test_30s_returns_60(self):
        assert auto_fps_focus(30) == 60

    def test_45s_returns_60(self):
        assert auto_fps_focus(45) == 60

    def test_60s_returns_60(self):
        assert auto_fps_focus(60) == 60

    def test_120s_returns_80(self):
        assert auto_fps_focus(120) == 80

    def test_large_returns_80(self):
        assert auto_fps_focus(600) == 80

    def test_negative_returns_30(self):
        assert auto_fps_focus(-1) == 30


class TestExtractAtTimestamps:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_at_timestamps("nonexistent.mp4", [0.0, 1.0])

    def test_returns_correct_structure(self):
        """Integration test - needs a real video file."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        frames = extract_at_timestamps(str(videos[0]), [0.0, 1.0])
        assert len(frames) > 0
        for f in frames:
            assert "index" in f
            assert "timestamp" in f
            assert "timestamp_str" in f
            assert "image" in f
            assert f["reason"] == "timestamp-cue"
            assert f["pinned"] is True

    def test_empty_timestamps(self):
        """Integration test with empty list."""
        import os
        test_data = Path("test_data")
        if not test_data.exists():
            pytest.skip("No test data available")
        videos = list(test_data.glob("*.mp4")) + list(test_data.glob("*.avi"))
        if not videos:
            pytest.skip("No test video available")

        frames = extract_at_timestamps(str(videos[0]), [])
        assert frames == []
