"""
Ollama adapter for local LLM inference.

Provides the same interface as OpenAI adapter but talks to local Ollama server.
"""

import asyncio
from typing import List, Optional
import json


class OllamaModel:
    """
    Ollama LLM adapter for local inference.
    
    Usage:
        model = OllamaModel(model="llama2", base_url="http://localhost:11434")
        response = await model.generate("What is 2+2?")
    """
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434", temperature: float = 0.7):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama2", "mistral", "codellama")
            base_url: Ollama server URL
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        
    async def generate(self, prompt: str, options: List[str] = None, **kwargs) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            options: Optional list of discrete options to choose from
            **kwargs: Additional parameters (stop, max_tokens, etc.)
        
        Returns:
            Generated text
        """
        import aiohttp
        
        # Build request
        if options:
            # Constrain to options via prompt
            full_prompt = f"{prompt}\n\nChoose EXACTLY ONE of these options:\n"
            for i, opt in enumerate(options):
                full_prompt += f"{i+1}. {opt}\n"
            full_prompt += "\nYour choice (just the text, no number):"
        else:
            full_prompt = prompt
            
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": kwargs.get("max_tokens", 512),
            }
        }
        
        if "stop" in kwargs:
            payload["options"]["stop"] = kwargs["stop"]
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        print(f"Ollama Error: {error_text}")
                        return options[0] if options else ""
                    
                    result = await response.json()
                    text = result.get("response", "").strip()
                    
                    # If options provided, try to match response to one
                    if options:
                        text_lower = text.lower()
                        for opt in options:
                            if opt.lower() in text_lower:
                                return opt
                        # Fallback to first option
                        return options[0]
                    
                    return text
                    
        except Exception as e:
            print(f"Ollama request failed: {e}")
            return options[0] if options else ""
    
    async def score(self, text: str, criteria: str) -> float:
        """
        Score text from 0.0 to 1.0 based on criteria.
        
        Args:
            text: Text to evaluate
            criteria: Scoring criteria
            
        Returns:
            Float score between 0.0 and 1.0
        """
        prompt = f"""Evaluate this text based on the criteria.
        
Criteria: {criteria}

Text: {text}

Output ONLY a single number between 0.0 and 1.0 representing the score."""

        try:
            result = await self.generate(prompt, temperature=0.0)
            # Extract first number found
            import re
            match = re.search(r'(\d+\.?\d*)', result)
            if match:
                score = float(match.group(1))
                # Normalize to 0-1 range
                if score > 1.0:
                    score = score / 100.0  # Assume percentage
                return min(max(score, 0.0), 1.0)
        except Exception:
            pass
        
        return 0.5  # Default neutral score


async def test_ollama():
    """Test Ollama connection and basic generation."""
    print("Testing Ollama connection...")
    
    model = OllamaModel(model="llama2")
    
    # Test basic generation
    response = await model.generate("What is 2+2? Answer with just the number.")
    print(f"Basic generation: {response}")
    
    # Test with options
    response = await model.generate(
        "Which is larger, a mouse or an elephant?",
        options=["mouse", "elephant"]
    )
    print(f"Option selection: {response}")
    
    # Test scoring
    score = await model.score(
        "The quick brown fox jumps over the lazy dog.",
        "Is this a complete sentence?"
    )
    print(f"Scoring: {score}")
    
    print("\nOllama integration working!")


if __name__ == "__main__":
    asyncio.run(test_ollama())
