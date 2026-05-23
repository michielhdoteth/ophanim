"""Abstract provider base class for vision-language models."""
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VlmProvider(ABC):
    """Abstract base for all VLM providers (LM Studio, Ollama, etc.).

    Subclasses must implement describe_image, describe_frames, and check_health.
    The constructor receives a config dict with at minimum a "model" key.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "google/gemma-4-e2b")

    @abstractmethod
    def describe_image(
        self, image: np.ndarray, question: Optional[str] = None
    ) -> str:
        """Describe a single image. Optionally answer a specific question.

        Args:
            image: NumPy array in (H, W, C) RGB format.
            question: An optional question about the image.

        Returns:
            Text response from the VLM.
        """
        ...

    @abstractmethod
    def describe_frames(
        self,
        frames: list[tuple[float, np.ndarray]],
        question: Optional[str] = None,
    ) -> str:
        """Analyze multiple frames from a video timeline.

        Args:
            frames: List of (timestamp_seconds, image_array) tuples.
            question: An optional question about the frames.

        Returns:
            A summary or answer combining observations from all frames.
        """
        ...

    def query_text(self, prompt: str) -> str:
        """Send a text-only query to the VLM (no image).

        Useful for summarization, entity extraction, and other text-only tasks.
        Default implementation sends a 1x1 transparent image as fallback.
        Subclasses should override with a proper text-only API call when possible.

        Args:
            prompt: The text prompt to send.

        Returns:
            The model's text response.
        """
        # Default: send minimal image + text (most VLMs support this)
        import numpy as np
        dummy = np.zeros((1, 1, 3), dtype=np.uint8)
        return self.describe_image(dummy, prompt)

    @abstractmethod
    def check_health(self) -> bool:
        """Check if the provider is available and responsive.

        Returns:
            True if the provider is reachable and ready, False otherwise.
        """
        ...
