"""Standard library functions for EnCompass agents.

Provides decorators and utilities for controlling agent execution.
"""

from functools import wraps

from core.signals import Effect


def action(func):
    """
    Decorator that converts a function call into an Effect signal.

    When the decorated function is called, it yields an Effect(func, args, kwargs)
    instead of executing the function body. The ExecutionEngine will intercept
    this signal, execute the function (if not replaying), and inject the result.

    Usage:
        @action
        def my_side_effect(x):
            return x + 1

        def agent():
            result = yield my_side_effect(10)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # We yield the Effect. The generator must be iterated by the Engine.
        # Note: This means the agent function must be a generator and use 'yield'.
        return Effect(func=func, args=args, kwargs=kwargs)

    return wrapper


def kill_branch():
    """
    Signal to immediately terminate the current branch as a failure.
    """
    from core.signals import KillBranch

    return KillBranch()


def early_stop():
    """
    Signal to stop searching this branch (success).
    """
    from core.signals import EarlyStop

    return EarlyStop()


def record_costs(tokens: int = 0, dollars: float = 0.0):
    """
    Signal to record resource usage.
    """
    from core.signals import RecordCosts

    return RecordCosts(tokens=tokens, dollars=dollars)


class NoCopy:
    """Marker for objects that should not be deep-copied across branches."""


def protect(func, attempts=3, exceptions=(Exception,)):
    """
    Signal to execute a function with retries managed by the engine.

    Usage:
        result = yield protect(risky_func, attempts=3)
    """
    from core.signals import Protect

    def wrapper(*args, **kwargs):
        return Protect(
            func=func, args=args, kwargs=kwargs, attempts=attempts, exceptions=exceptions
        )

    return wrapper


@action
def calculator(expression: str) -> float:
    """
    A simple calculator tool.
    """
    try:
        # Safety: Use simple eval with limited scope, or just eval since it's local
        # For parity with paper, we assume a safe sandbox.
        return float(eval(expression, {"__builtins__": None}, {}))
    except Exception:
        return None


def local_search(strategy_cls, agent_factory, **kwargs):
    """
    Signal to run a nested search.
    Usage:
        result = yield local_search(BeamSearch, sub_agent, width=5)
    """
    from core.signals import LocalSearch

    # We pass a factory for the strategy because it needs the engine instance
    return LocalSearch(strategy_factory=strategy_cls, agent_factory=agent_factory, kwargs=kwargs)
