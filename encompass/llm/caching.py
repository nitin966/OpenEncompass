import hashlib
import json
import os
from typing import Any, cast

from core.llm import LanguageModel


class CachingLM:
    """
    A wrapper around any LanguageModel that caches responses based on inputs.
    Ensures deterministic replay across different process runs.
    """

    def __init__(self, base_lm: LanguageModel, cache_path: str = "llm_cache.json"):
        self.base_lm = base_lm
        self.cache_path = cache_path
        self.cache: dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save LLM cache: {e}")

    def _compute_key(self, method: str, prompt: str, **kwargs) -> str:
        data = {"method": method, "prompt": prompt, "kwargs": kwargs}
        # Sort keys for consistent hashing
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode()).hexdigest()

    async def generate(self, prompt: str, options: list[str] | None = None, **kwargs) -> str:
        key = self._compute_key("generate", prompt, options=options, **kwargs)

        if key in self.cache:
            return cast(str, self.cache[key])

        result = await self.base_lm.generate(prompt, options, **kwargs)
        self.cache[key] = result
        self._save_cache()
        return result

    async def score(self, text: str, criteria: str) -> float:
        key = self._compute_key("score", text, criteria=criteria)

        if key in self.cache:
            return cast(float, self.cache[key])

        result = await self.base_lm.score(text, criteria)
        self.cache[key] = result
        self._save_cache()
        return result
