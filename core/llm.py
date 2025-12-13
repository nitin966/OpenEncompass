"""Language Model Protocol and Mock Implementation."""

from typing import Protocol


class LanguageModel(Protocol):
    async def generate(self, prompt: str, options: list[str] | None = None) -> str:
        """
        Generate text based on prompt.
        If options are provided, return one of the options.
        """
        ...

    async def score(self, text: str, criteria: str) -> float:
        """
        Score the text based on criteria. Returns 0.0 to 1.0.
        """
        ...


class MockLLM:
    """Mock LLM for testing purposes."""

    async def generate(self, prompt: str, options: list[str] | None = None) -> str:
        """Generate mock response, selecting from options if provided."""
        # Simple heuristic or random choice for mock
        if options:
            # Deterministic mock: pick the longest option just to have logic
            # or just pick the first one.
            # Let's pick based on a hash of the prompt to be deterministic but varied
            import hashlib

            idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(options)
            return options[idx]
        return "Mock response"

    async def score(self, text: str, criteria: str) -> float:
        """Return mock score for text based on criteria."""
        # Mock scoring: length based or random
        return 0.8
