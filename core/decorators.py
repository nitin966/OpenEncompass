import functools
import inspect
from typing import Callable, Generator, Any
from core.compiler import compile_agent

def encompass_agent(func: Callable) -> Callable:
    # Compile the agent function into a state machine class
    # We do this at decoration time (or lazy load?)
    # Doing it at decoration time is better for errors.
    MachineClass = compile_agent(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        # Instantiate the compiled machine
        # The generated __init__ handles arguments
        return MachineClass(*args, **kwargs)
        
    return wrapper
