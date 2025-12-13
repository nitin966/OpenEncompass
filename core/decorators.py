"""Decorators for defining EnCompass agents."""

import functools
from collections.abc import Callable
from typing import Any

from core.compiler import compile_agent


def encompass_agent(func: Callable) -> Callable:
    """Compile a function into an EnCompass agent (state machine).

    Args:
        func: The function to compile into an agent

    Returns:
        A wrapper that creates instances of the compiled agent
    """
    # Compile the agent function into a state machine class
    # We do this at decoration time (or lazy load?)
    # Doing it at decoration time is better for errors.
    MachineClass = compile_agent(func)  # noqa: N806

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Instantiate the compiled machine
        # The generated __init__ handles arguments
        return MachineClass(*args, **kwargs)

    return wrapper
