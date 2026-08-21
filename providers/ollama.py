"""Ollama VLM provider using native /api/chat endpoint.

Ollama uses a different API than OpenAI-compatible endpoints.
This provider uses Ollama's native /api/chat with vision support.
"""
from typing import Optional

import httpx
import numpy as np

from providers.base import VlmProvider, VlmResponse, TokenUsage
from core.image import encode_base64


class OllamaProvider(VlmProvider):
    """Ollama provider using native /api/chat endpoint with vision support.

    Connects to a local Ollama instance at ``base_url`` (default
    ``http://localhost:11434``) and uses its /api/chat endpoint
    with ``images`` field carrying base64-encoded images.

    Supports:
        - Vision models (llava, bakllava, etc.)
        - keep_alive for model persistence
        - Auto-pull model if not available
    """

    PROVIDER_NAME = "ollama"

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1024)
        self.timeout = config.get("timeout", 120)
        self.keep_alive = config.get("keep_alive", "5m")
        self._client = httpx.Client(timeout=self.timeout)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _chat_completion(self, messages: list[dict]) -> dict:
        """Send a chat completion request via Ollama native API."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "keep_alive": self.keep_alive,
        }
        response = self._client.post(
            f"{self.base_url}/api/chat",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def _error_response(self, exc: Exception, context: str = "Ollama") -> VlmResponse:
        """Create a VlmResponse from an exception."""
        if isinstance(exc, httpx.ConnectError):
            return VlmResponse(
                content=(
                    f"Could not connect to {context} at {self.base_url}: {exc}. "
                    "Is Ollama running? Start it with: ollama serve"
                ),
                tokens=TokenUsage(),
            )
        elif isinstance(exc, httpx.TimeoutException):
            return VlmResponse(
                content=(
                    f"Request to {context} timed out after {self.timeout}s. "
                    "The model may be loading or the request is too large."
                ),
                tokens=TokenUsage(),
            )
        elif isinstance(exc, httpx.HTTPStatusError):
            return VlmResponse(
                content=(
                    f"{context} returned HTTP {exc.response.status_code}: "
                    f"{exc.response.text[:200]}"
                ),
                tokens=TokenUsage(),
            )
        else:
            return VlmResponse(
                content=f"Unexpected error contacting {context}: {exc}",
                tokens=TokenUsage(),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_image(
        self, image: np.ndarray, question: Optional[str] = None
    ) -> VlmResponse:
        """Send an image to Ollama and get a description."""
        b64_image = encode_base64(image)
        # Ollama expects raw base64 without the data URI prefix
        if b64_image.startswith("data:"):
            b64_image = b64_image.split(",", 1)[1]

        if question:
            text = (
                f"Answer this question about the image: {question}\n"
                "Be concise and specific. Read all numbers, labels, and text carefully."
            )
        else:
            text = (
                "Describe this image in detail. "
                "If it contains a chart, graph, or data visualization, "
                "read ALL numbers, labels, axis values, and data points precisely. "
                "List the main objects visible."
            )

        messages = [
            {
                "role": "user",
                "content": text,
                "images": [b64_image],
            }
        ]

        try:
            data = self._chat_completion(messages)
            content = data.get("message", {}).get("content", "").strip()
            # Ollama returns eval_count/eval_duration for token info
            tokens = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            )
            return VlmResponse(content=content, tokens=tokens)
        except Exception as exc:
            return self._error_response(exc)

    def describe_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        question: Optional[str] = None,
    ) -> str:
        """Analyze multiple frames from a video timeline."""
        if not frames:
            return "No frames to analyze."

        results: list[str] = []
        for timestamp, image in frames[:10]:
            resp = self.describe_image(image, question)
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            results.append(f"[{mins:02d}:{secs:02d}] {resp.content}")

        if len(frames) > 10:
            results.append(
                f"... and {len(frames) - 10} more frames (truncated)"
            )

        return "\n".join(results)

    def query_text(self, prompt: str) -> VlmResponse:
        """Send a text-only query to Ollama (no image)."""
        messages = [{"role": "user", "content": prompt}]
        try:
            data = self._chat_completion(messages)
            content = data.get("message", {}).get("content", "").strip()
            tokens = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            )
            return VlmResponse(content=content, tokens=tokens)
        except Exception as exc:
            return self._error_response(exc)

    def check_health(self) -> bool:
        """Check if Ollama API is responsive."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models from Ollama."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model_name: Model to pull (default: self.model_name).

        Returns:
            True if pull succeeded, False otherwise.
        """
        target = model_name or self.model_name
        try:
            response = self._client.post(
                f"{self.base_url}/api/pull",
                json={"name": target, "stream": False},
                timeout=300,  # Models can be large
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
