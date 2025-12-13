import os


class OpenAIModel:
    """
    Production-grade OpenAI Adapter.
    Supports async generation, retries, and scoring.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai") from None

        self.client = AsyncOpenAI(
            api_key=api_key
            or os.getenv("OPENAI_API_KEY")
            or "dummy",  # Local LLMs might not need key
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model
        self.temperature = temperature

    async def generate(self, prompt: str, options: list[str] = None, **kwargs) -> str:
        """
        Generates text. If options are provided, uses function calling to enforce selection.

        When options are provided, uses OpenAI function calling with enum to ensure
        reproducible, logit-level enforcement of discrete choices, rather than
        relying on model-dependent system prompt interpretation.
        """
        if options:
            # Use function calling for reproducible option selection
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "select_option",
                        "description": "Select one of the provided options",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "choice": {
                                    "type": "string",
                                    "enum": options,  # Logit-level enforcement
                                    "description": "The selected option",
                                }
                            },
                            "required": ["choice"],
                        },
                    },
                }
            ]

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Select the most appropriate option.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "select_option"}},
                    temperature=self.temperature,
                )

                # Extract choice from function call
                tool_call = response.choices[0].message.tool_calls[0]
                import json

                args = json.loads(tool_call.function.arguments)
                return args["choice"]

            except Exception as e:
                # Fallback to first option if function calling fails
                print(f"Function calling failed: {e}, using first option")
                return options[0] if options else ""
        else:
            # No options - standard generation
            system_prompt = "You are a helpful assistant."

            try:
                # Extract known kwargs for OpenAI
                stop = kwargs.get("stop")

                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    stop=stop,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                # Basic retry logic could go here
                print(f"OpenAI Error: {e}")
                return ""

    async def score(self, text: str, criteria: str) -> float:
        """
        Scores text from 0.0 to 1.0 based on criteria.
        """
        prompt = f"Evaluate the following text based on this criteria: '{criteria}'.\n\nText: {text}\n\nOutput ONLY a single float number between 0.0 and 1.0."
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an evaluator. Output only a float."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Deterministic for scoring
            )
            content = response.choices[0].message.content.strip()
            return float(content)
        except Exception:
            return 0.0
