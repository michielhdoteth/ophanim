"""Tests for VLM providers."""
import pytest
import numpy as np

from openvision.providers.base import VlmProvider, VlmResponse, TokenUsage
from openvision.providers.lmstudio import LmStudioProvider
from openvision.providers.openai_compat import OpenAICompatProvider
from openvision.providers.ollama import OllamaProvider
from openvision.providers.llamacpp import LlamaCppProvider
from openvision.providers.cloud import CloudProvider
from openvision.providers.registry import ProviderRegistry, _extract_models


# ------------------------------------------------------------------
# Abstract base class
# ------------------------------------------------------------------

class TestVlmProviderBase:
    def test_cannot_instantiate_abstract(self):
        """VlmProvider is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VlmProvider({"model": "test"})  # type: ignore[abstract]


# ------------------------------------------------------------------
# TokenUsage / VlmResponse dataclasses
# ------------------------------------------------------------------

class TestTokenUsage:
    def test_defaults(self):
        t = TokenUsage()
        assert t.prompt_tokens == 0
        assert t.completion_tokens == 0
        assert t.reasoning_tokens == 0
        assert t.total_tokens == 0

    def test_custom_values(self):
        t = TokenUsage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=5, total_tokens=30)
        assert t.prompt_tokens == 10
        assert t.completion_tokens == 20
        assert t.reasoning_tokens == 5
        assert t.total_tokens == 30


class TestVlmResponse:
    def test_defaults(self):
        r = VlmResponse(content="hello")
        assert r.content == "hello"
        assert isinstance(r.tokens, TokenUsage)

    def test_with_tokens(self):
        tokens = TokenUsage(prompt_tokens=5)
        r = VlmResponse(content="world", tokens=tokens)
        assert r.tokens.prompt_tokens == 5


# ------------------------------------------------------------------
# OpenAICompatProvider (shared base)
# ------------------------------------------------------------------

class TestOpenAICompatProvider:
    """Tests for the shared OpenAI-compatible base class."""

    def test_init(self):
        config = {
            "model": "test-model",
            "base_url": "http://localhost:1234/v1",
            "temperature": 0.5,
            "max_tokens": 256,
            "timeout": 30,
        }
        provider = OpenAICompatProvider(config)
        try:
            assert provider.model_name == "test-model"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 256
            assert provider.timeout == 30
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = OpenAICompatProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 1024
            assert provider.timeout == 60
        finally:
            provider.close()

    def test_api_key_sets_auth_header(self):
        config = {"api_key": "sk-test123"}
        provider = OpenAICompatProvider(config)
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-test123"
        finally:
            provider.close()

    def test_no_api_key_no_auth_header(self):
        config = {"api_key": "not-needed"}
        provider = OpenAICompatProvider(config)
        try:
            assert "Authorization" not in provider._extra_headers
        finally:
            provider.close()

    def test_parse_usage(self):
        data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 10},
            }
        }
        usage = OpenAICompatProvider._parse_usage(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.reasoning_tokens == 10
        assert usage.total_tokens == 150

    def test_parse_usage_missing_fields(self):
        usage = OpenAICompatProvider._parse_usage({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_describe_frames_empty(self):
        provider = OpenAICompatProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
        """More than 10 frames gets truncated."""
        config = {
            "base_url": "http://localhost:19999/v1",
            "model": "test",
            "timeout": 2,
        }
        provider = OpenAICompatProvider(config)
        try:
            frames = [
                (float(i), np.ones((10, 10, 3), dtype=np.uint8) * 128)
                for i in range(15)
            ]
            result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11  # 10 frames + 1 truncation line
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    def test_describe_frames_timestamps(self):
        """Frames are formatted as [MM:SS] in output."""
        config = {
            "base_url": "http://localhost:19999/v1",
            "model": "test",
            "timeout": 2,
        }
        provider = OpenAICompatProvider(config)
        try:
            frames = [(65.0, np.ones((10, 10, 3), dtype=np.uint8) * 128)]
            result = provider.describe_frames(frames, "test")
            assert "[01:05]" in result
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
        }
        provider = OpenAICompatProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = OpenAICompatProvider({"base_url": "http://localhost:19999/v1"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = OpenAICompatProvider({"base_url": "http://localhost:19999/v1"})
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()


# ------------------------------------------------------------------
# LM Studio provider
# ------------------------------------------------------------------

class TestLmStudioProvider:
    """Unit / integration tests for LmStudioProvider."""

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
            assert provider.PROVIDER_NAME == "lmstudio"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = LmStudioProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:1234/v1"
        finally:
            provider.close()

    def test_check_health_lmstudio_running(self):
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

    def test_describe_image(self):
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
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_describe_image_with_question(self):
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
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_describe_frames(self):
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
        provider = LmStudioProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
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
            result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "google/gemma-4-e2b",
        }
        provider = LmStudioProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = LmStudioProvider({"base_url": "http://localhost:19999/v1"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()


# ------------------------------------------------------------------
# Ollama provider
# ------------------------------------------------------------------

class TestOllamaProvider:
    """Tests for the Ollama native API provider."""

    def test_init(self):
        config = {
            "model": "llava:13b",
            "base_url": "http://localhost:11434",
            "temperature": 0.3,
            "max_tokens": 2048,
            "timeout": 60,
            "keep_alive": "10m",
        }
        provider = OllamaProvider(config)
        try:
            assert provider.model_name == "llava:13b"
            assert provider.base_url == "http://localhost:11434"
            assert provider.temperature == 0.3
            assert provider.max_tokens == 2048
            assert provider.timeout == 60
            assert provider.keep_alive == "10m"
            assert provider.PROVIDER_NAME == "ollama"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = OllamaProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:11434"
            assert provider.temperature == 0.1
            assert provider.max_tokens == 1024
            assert provider.timeout == 120
            assert provider.keep_alive == "5m"
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = OllamaProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_describe_frames_truncated(self):
        config = {
            "base_url": "http://localhost:19999",
            "model": "test",
            "timeout": 2,
        }
        provider = OllamaProvider(config)
        try:
            frames = [
                (float(i), np.ones((10, 10, 3), dtype=np.uint8) * 128)
                for i in range(15)
            ]
            result = provider.describe_frames(frames, "test")
            lines = result.split("\n")
            assert len(lines) == 11
            assert any("truncated" in line for line in lines)
        finally:
            provider.close()

    def test_describe_frames_timestamps(self):
        config = {
            "base_url": "http://localhost:19999",
            "model": "test",
            "timeout": 2,
        }
        provider = OllamaProvider(config)
        try:
            frames = [(65.0, np.ones((10, 10, 3), dtype=np.uint8) * 128)]
            result = provider.describe_frames(frames, "test")
            assert "[01:05]" in result
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999",
            "model": "test",
        }
        provider = OllamaProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
            assert "ollama serve" in result.content
        finally:
            provider.close()

    def test_connection_error_text_query(self):
        config = {
            "base_url": "http://localhost:9999",
            "model": "test",
        }
        provider = OllamaProvider(config)
        try:
            result = provider.query_text("hello")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = OllamaProvider({"base_url": "http://localhost:19999"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_health_check_ollama_running(self):
        """Integration test - Ollama should be reachable."""
        provider = OllamaProvider({})
        try:
            health = provider.check_health()
            assert isinstance(health, bool)
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = OllamaProvider({"base_url": "http://localhost:19999"})
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()

    def test_list_models_ollama_running(self):
        """Integration test - list models from Ollama."""
        provider = OllamaProvider({})
        try:
            if not provider.check_health():
                pytest.skip("Ollama not running")
            models = provider.list_models()
            assert isinstance(models, list)
        finally:
            provider.close()

    def test_describe_image(self):
        """Integration test - describe an image via Ollama."""
        config = {"model": "llava:7b"}
        provider = OllamaProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("Ollama not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img)
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_describe_image_with_question(self):
        """Integration test - describe image with specific question."""
        config = {"model": "llava:7b"}
        provider = OllamaProvider(config)
        try:
            if not provider.check_health():
                pytest.skip("Ollama not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img, "What color is this?")
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_query_text(self):
        """Integration test - text-only query to Ollama."""
        provider = OllamaProvider({})
        try:
            if not provider.check_health():
                pytest.skip("Ollama not running")
            result = provider.query_text("Say hello in one word.")
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_pull_model_bad_url(self):
        provider = OllamaProvider({"base_url": "http://localhost:19999"})
        try:
            result = provider.pull_model("nonexistent")
            assert result is False
        finally:
            provider.close()


# ------------------------------------------------------------------
# llama.cpp provider
# ------------------------------------------------------------------

class TestLlamaCppProvider:
    """Tests for the llama.cpp provider (thin wrapper over OpenAICompatProvider)."""

    def test_init(self):
        config = {
            "model": "llava-13b",
            "base_url": "http://localhost:8080/v1",
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        provider = LlamaCppProvider(config)
        try:
            assert provider.model_name == "llava-13b"
            assert provider.base_url == "http://localhost:8080/v1"
            assert provider.temperature == 0.2
            assert provider.max_tokens == 1024
            assert provider.PROVIDER_NAME == "llamacpp"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = LlamaCppProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            assert provider.base_url == "http://localhost:8080/v1"
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = LlamaCppProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
        }
        provider = LlamaCppProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = LlamaCppProvider({"base_url": "http://localhost:19999/v1"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_health_check_llamacpp_running(self):
        """Integration test - llama.cpp should be reachable."""
        provider = LlamaCppProvider({})
        try:
            health = provider.check_health()
            assert isinstance(health, bool)
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = LlamaCppProvider({"base_url": "http://localhost:19999/v1"})
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()

    def test_describe_image(self):
        """Integration test - describe an image via llama.cpp."""
        provider = LlamaCppProvider({})
        try:
            if not provider.check_health():
                pytest.skip("llama.cpp not running")
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            result = provider.describe_image(img)
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()

    def test_query_text(self):
        """Integration test - text-only query to llama.cpp."""
        provider = LlamaCppProvider({})
        try:
            if not provider.check_health():
                pytest.skip("llama.cpp not running")
            result = provider.query_text("Say hello.")
            assert isinstance(result, VlmResponse)
            assert len(result.content) > 0
        finally:
            provider.close()


# ------------------------------------------------------------------
# Cloud provider (OpenAI, Groq, Together, vLLM, etc.)
# ------------------------------------------------------------------

class TestCloudProvider:
    """Tests for the generic OpenAI-compatible provider (OpenAI, Groq, vLLM, etc.)."""

    def test_init(self):
        config = {
            "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "temperature": 0.5,
            "max_tokens": 2048,
        }
        provider = CloudProvider(config)
        try:
            assert provider.model_name == "gpt-4o"
            assert provider.base_url == "https://api.openai.com/v1"
            assert provider.temperature == 0.5
            assert provider.max_tokens == 2048
            assert provider.PROVIDER_NAME == "openai"
        finally:
            provider.close()

    def test_init_defaults(self):
        provider = CloudProvider({})
        try:
            assert provider.model_name == "google/gemma-4-e2b"
            # Falls back to "not-needed" when no env var set
            assert provider._extra_headers.get("Authorization") is None
        finally:
            provider.close()

    def test_env_var_fallback(self, monkeypatch):
        """API key falls back to OPENAI_API_KEY env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        provider = CloudProvider({"model": "test"})
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-env-test"
        finally:
            provider.close()

    def test_config_key_takes_precedence(self, monkeypatch):
        """Config api_key takes precedence over env var."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-test")
        provider = CloudProvider({"api_key": "sk-config-test"})
        try:
            assert provider._extra_headers["Authorization"] == "Bearer sk-config-test"
        finally:
            provider.close()

    def test_describe_frames_empty(self):
        provider = CloudProvider({})
        try:
            result = provider.describe_frames([])
            assert result == "No frames to analyze."
        finally:
            provider.close()

    def test_connection_error_returns_message(self):
        config = {
            "base_url": "http://localhost:9999/v1",
            "model": "test",
        }
        provider = CloudProvider(config)
        try:
            img = np.ones((10, 10, 3), dtype=np.uint8)
            result = provider.describe_image(img, "test")
            assert isinstance(result, VlmResponse)
            assert "Could not connect" in result.content
        finally:
            provider.close()

    def test_health_check_bad_url(self):
        provider = CloudProvider({"base_url": "http://localhost:19999/v1"})
        try:
            assert provider.check_health() is False
        finally:
            provider.close()

    def test_list_models_bad_url(self):
        provider = CloudProvider({"base_url": "http://localhost:19999/v1"})
        try:
            models = provider.list_models()
            assert models == []
        finally:
            provider.close()


# ------------------------------------------------------------------
# ProviderRegistry
# ------------------------------------------------------------------

class TestProviderRegistry:
    """Tests for the provider registry and factory."""

    def test_create_explicit_lmstudio(self):
        config = {"provider": "lmstudio", "model": "test-model"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, LmStudioProvider)
            assert provider.model_name == "test-model"
        finally:
            provider.close()

    def test_create_explicit_ollama(self):
        config = {"provider": "ollama", "model": "llava:7b"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, OllamaProvider)
            assert provider.model_name == "llava:7b"
        finally:
            provider.close()

    def test_create_explicit_llamacpp(self):
        config = {"provider": "llamacpp", "model": "llava"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, LlamaCppProvider)
            assert provider.model_name == "llava"
        finally:
            provider.close()

    def test_create_explicit_openai(self):
        config = {"provider": "openai", "model": "gpt-4o", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
            assert provider.model_name == "gpt-4o"
        finally:
            provider.close()

    def test_create_explicit_groq(self):
        config = {"provider": "groq", "model": "llava", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_explicit_together(self):
        config = {"provider": "together", "model": "llava", "api_key": "sk-test"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_explicit_vllm(self):
        config = {"provider": "vllm", "model": "llava", "api_key": "not-needed"}
        provider = ProviderRegistry.create(config)
        try:
            assert isinstance(provider, CloudProvider)
        finally:
            provider.close()

    def test_create_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderRegistry.create({"provider": "nonexistent"})

    def test_create_auto_with_mock(self, monkeypatch):
        """Auto-detect falls back to first detected provider."""
        # Mock detect to return a known provider
        monkeypatch.setattr(
            ProviderRegistry, "detect",
            lambda: [{"name": "lmstudio", "config": {"base_url": "http://test", "model": "m"}}],
        )
        provider = ProviderRegistry.create({"provider": "auto"})
        try:
            assert isinstance(provider, LmStudioProvider)
        finally:
            provider.close()

    def test_create_auto_no_providers(self, monkeypatch):
        """Auto-detect with no running providers raises ValueError."""
        monkeypatch.setattr(ProviderRegistry, "detect", lambda: [])
        with pytest.raises(ValueError, match="No VLM provider found"):
            ProviderRegistry.create({"provider": "auto"})

    def test_list_providers(self):
        providers = ProviderRegistry.list_providers()
        assert "lmstudio" in providers
        assert "ollama" in providers
        assert "llamacpp" in providers
        assert "openai" in providers
        assert "groq" in providers

    def test_detect_returns_list(self, monkeypatch):
        """detect() returns a list even with mocked endpoints."""
        import httpx
        def mock_get(url, **kwargs):
            raise httpx.ConnectError("mock")
        monkeypatch.setattr(httpx, "get", mock_get)
        result = ProviderRegistry.detect()
        assert isinstance(result, list)
        assert len(result) >= 3

    def test_detect_result_structure(self, monkeypatch):
        import httpx
        def mock_get(url, **kwargs):
            raise httpx.ConnectError("mock")
        monkeypatch.setattr(httpx, "get", mock_get)
        result = ProviderRegistry.detect()
        for item in result:
            assert "name" in item
            assert "status" in item
            assert item["status"] in ("available", "unavailable", "timeout", "error")

    def test_lmstudio_defaults_set(self):
        config = {"provider": "lmstudio"}
        provider = ProviderRegistry.create(config)
        try:
            assert provider.base_url == "http://localhost:1234/v1"
        finally:
            provider.close()

    def test_ollama_defaults_set(self):
        config = {"provider": "ollama"}
        provider = ProviderRegistry.create(config)
        try:
            assert provider.base_url == "http://localhost:11434"
        finally:
            provider.close()

    def test_llamacpp_defaults_set(self):
        config = {"provider": "llamacpp"}
        provider = ProviderRegistry.create(config)
        try:
            assert provider.base_url == "http://localhost:8080/v1"
        finally:
            provider.close()


# ------------------------------------------------------------------
# _extract_models helper
# ------------------------------------------------------------------

class TestExtractModels:
    def test_ollama_format(self):
        data = {"models": [{"name": "llava:7b"}, {"name": "mistral"}]}
        models = _extract_models("ollama", data)
        assert models == ["llava:7b", "mistral"]

    def test_lmstudio_format(self):
        data = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
        models = _extract_models("lmstudio", data)
        assert models == ["model-a", "model-b"]

    def test_llamacpp_format(self):
        data = {"data": [{"id": "ggml-model"}]}
        models = _extract_models("llamacpp", data)
        assert models == ["ggml-model"]

    def test_unknown_provider(self):
        models = _extract_models("unknown", {})
        assert models == []

    def test_empty_data(self):
        assert _extract_models("ollama", {}) == []
        assert _extract_models("lmstudio", {}) == []
