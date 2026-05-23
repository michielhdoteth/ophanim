"""LM Studio VLM provider via OpenAI-compatible HTTP API."""
from typing import Optional

import httpx
import numpy as np

from ophanim.providers.base import VlmProvider
from ophanim.core.image import encode_base64


class LmStudioProvider(VlmProvider):
    """LM Studio provider using the OpenAI-compatible chat completions endpoint.

    Connects to a local LM Studio instance at ``base_url`` (default
    ``http://localhost:1234/v1``) and uses its /v1/chat/completions endpoint
    with ``image_url`` content blocks that carry base64-encoded images.

    All HTTP calls go through a single reusable ``httpx.Client`` instance.
    Call ``.close()`` when done to release the connection pool.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:1234/v1")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 512)
        self.timeout = config.get("timeout", 60)
        self._client = httpx.Client(timeout=self.timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_image(
        self, image: np.ndarray, question: Optional[str] = None
    ) -> str:
        """Send an image to LM Studio and get a description.

        Args:
            image: NumPy array in (H, W, C) RGB format.
            question: Optional question to ask about the image.

        Returns:
            The model's text response, or an error message on failure.
        """
        b64_image = encode_base64(image)

        if question:
            text = (
                f"Answer this question about the image: {question}\n"
                "Be concise and specific."
            )
        else:
            text = (
                "Describe this image in 2-3 sentences. "
                "List the main objects visible."
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": b64_image},
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except httpx.ConnectError as exc:
            return (
                f"Could not connect to LM Studio at {self.base_url}: {exc}. "
                "Is the server running?"
            )
        except httpx.TimeoutException:
            return (
                f"Request to LM Studio timed out after {self.timeout}s. "
                "The model may be overloaded."
            )
        except httpx.HTTPStatusError as exc:
            return (
                f"LM Studio returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:200]}"
            )
        except Exception as exc:
            return f"Unexpected error contacting LM Studio: {exc}"

    def describe_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        question: Optional[str] = None,
    ) -> str:
        """Analyze multiple frames.

        If more than 10 frames are provided only the first 10 are processed
        and a truncation notice is appended.  Each frame is described
        independently and the results are concatenated with timestamps.

        Args:
            frames: List of ``(timestamp_seconds, image_array)`` tuples.
            question: Optional question to ask for each frame.

        Returns:
            Combined text response from all processed frames.
        """
        if not frames:
            return "No frames to analyze."

        results: list[str] = []
        for timestamp, image in frames[:10]:
            result = self.describe_image(image, question)
            mins = int(timestamp // 60)
            secs = int(timestamp % 60)
            results.append(f"[{mins:02d}:{secs:02d}] {result}")

        if len(frames) > 10:
            results.append(
                f"... and {len(frames) - 10} more frames (truncated)"
            )

        return "\n".join(results)

    def query_text(self, prompt: str) -> str:
        """Send a text-only query to LM Studio (no image).

        LM Studio's chat API supports text-only messages.
        This avoids sending a dummy image for summarization tasks.
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[VLM text query failed: {e}]"

    def check_health(self) -> bool:
        """Check if LM Studio API is responsive.

        Hits the ``/v1/models`` endpoint with a short 5-second timeout.
        Returns ``False`` instead of raising on any error.
        """
        try:
            response = self._client.get(f"{self.base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        self._client.close()
