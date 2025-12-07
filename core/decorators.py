import functools
import inspect
from typing import Callable, Generator

def encompass_agent(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Generator:
        gen = func(*args, **kwargs)
        if not inspect.isgenerator(gen):
            raise TypeError(f"Agent {func.__name__} must return a generator.")
        return gen
    return wrapper
