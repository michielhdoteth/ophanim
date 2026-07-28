"""Cloud VLM provider — remote API backends.

Works with OpenAI, Groq, vLLM, Together AI, LocalAI, and any other
provider that implements the OpenAI chat completions protocol with vision.
"""
import os
from typing import Optional

from openvision.providers.openai_compat import OpenAICompatProvider


class CloudProvider(OpenAICompatProvider):
    """Remote/cloud provider for any OpenAI-compatible API with vision support.

    Supports: OpenAI, Groq, vLLM, Together AI, LocalAI, Anyscale, etc.

    Config keys consumed:
        base_url: API base URL (required for non-OpenAI providers)
        model: Model identifier (required)
        api_key: API key (or env var OPENAI_API_KEY)
        temperature: Sampling temperature (default: 0.1)
        max_tokens: Max response tokens (default: 1024)
        timeout: HTTP timeout in seconds (default: 60)
    """

    PROVIDER_NAME = "openai"

    def __init__(self, config: dict):
        # Resolve API key: config → env var
        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = "not-needed"  # Allow unauthenticated local servers

        config["api_key"] = api_key
        super().__init__(config)


# Backward compat alias
GenericOpenAIProvider = CloudProvider
