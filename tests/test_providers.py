"""Tests for VLM providers."""
import pytest
import numpy as np

from ophanim.providers.base import VlmProvider
from ophanim.providers.lmstudio import LmStudioProvider


# ------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------

class TestVlmProviderBase:
    def test_cannot_instantiate_abstract(self):
        """VlmProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VlmProvider({"model": "test"})  # type: ignore[abstract]


# ------------------------------------------------------------------
# LM Studio provider
# ------------------------------------------------------------------

class TestLmStudioProvider:
    """Unit / integration tests for LmStudioProvider.

    Tests that actually call the LM Studio API are gated behind
    ``provider.check_health()`` so they automatically skip when the
    server is not running.
    """

    # -- unit-level ---------------------------------------------------

    def test_init(self):
        config = {
            "model": "google/gemma-4-e2b",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.1,
            "max_tokens": 512,
        }
        provider = LmStudioProvider(config)
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 512
        finally:
            provider.close()

    def test_init_defaults(self):
        """Config fields fall back to sensible defaults."""
        provider = LmStudioProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
        finally:
            provider.close()

    # -- health check -------------------------------------------------

    def test_check_health_lmstudio_running(self):
        """Integration test - LM Studio should be reachable."""
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            health = provider.check_health()
            assert isinstance(health, bool)
        finally:
            provider.close()

    # -- image description --------------------------------------------

    def test_describe_image(self):
        """Describe a simple solid-colour image."""
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")

            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img)
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            provider.close()

    def test_describe_image_with_question(self):
        """Describe an image with a specific question."""
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")

            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img, "What color is this?")
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            provider.close()

    # -- frame analysis -----------------------------------------------

    def test_describe_frames(self):
        """Describe a single frame (list of one)."""
        config = {
            "base_url": "http://localhost:1234/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("LM Studio not running")

            frames = [(0.0, np.ones((100, 100, 3), dtype=np.uint8) * 128)]
            result = provider.describe_frames(frames, "What color is this?")
            assert isinstance(result, str)
            assert len(result) > 0
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        """Empty frame list returns a no-op message."""
        provider = LmStudioProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
        """More than 10 frames gets truncated with a notice."""
        config = {
            "base_url": "http://localhost:19999/v1",
            "model": "google/gemma-4-e2b",
            "timeout": 2,
        }
        provider = LmStudioProvider(config)
        try:
            frames = [
                (float(i), np.ones((10, 10, 3), dtype=np.uint8) * 128)
                for i in range(15)
            ]

            # Use a bad URL so each describe_image fails fast with a
            # connection error.  We verify the truncation logic by
            # checking the last line contains "truncated".
            result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11  # 10 frames + 1 truncation line
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    # -- error resilience ---------------------------------------------

    def test_connection_error_returns_message(self):
        """When LM Studio is unreachable, describe_image returns an
        informative message instead of crashing."""
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, str)
            assert "Could not connect" in result
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        """check_health returns False when the server does not exist."""
        provider = LmStudioProvider({"base_url": "http://localhost:19999/v1"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()
