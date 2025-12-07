from dataclasses import dataclass, field
from typing import Dict, Any, Union, Callable, List

@dataclass(frozen=True)
class ControlSignal:
    """Base class for all control signals yielded by agents."""
    pass

@dataclass(frozen=True)
class BranchPoint(ControlSignal):
    """
    Signal to indicate a branching point in the execution.
    
    The engine will use the sampler to choose one of the options.
    
    Attributes:
        name: A unique identifier for this branch point.
        metadata: Optional dictionary containing context, options, or schema for the decision.
    """
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class Effect(ControlSignal):
    """
    Signal indicating a side-effect or IO operation that should be replayed.
    
    Attributes:
        func: The callable to execute.
        args: Positional arguments.
        kwargs: Keyword arguments.
        key: Optional unique key for caching (if not provided, derived from args).
    """
    func: Callable
    args: tuple
    kwargs: Dict[str, Any]
    key: Union[str, None] = None

@dataclass(frozen=True)
class ScoreSignal(ControlSignal):
    """
    Signal indicating an intermediate or final reward/score.
    
    Attributes:
        value: The numerical score value.
        context: Optional description of what is being scored.
    """
    value: float
    context: str = ""

@dataclass(frozen=True)
class Protect(ControlSignal):
    """
    Signal to protect a function call with retries/resampling.
    """
    func: Callable
    args: tuple
    kwargs: Dict[str, Any]
    attempts: int
    exceptions: tuple

@dataclass(frozen=True)
class EarlyStop(ControlSignal):
    """Signal to stop searching this branch (success)."""
    pass

@dataclass(frozen=True)
class KillBranch(ControlSignal):
    """Signal to prune this branch (failure)."""
    pass

@dataclass(frozen=True)
class RecordCosts(ControlSignal):
    """Signal to record resource usage."""
    tokens: int = 0
    dollars: float = 0.0

@dataclass(frozen=True)
class LocalSearch(ControlSignal):
    """
    Signal to run a nested search strategy.
    """
    strategy_factory: Callable
    agent_factory: Callable
    kwargs: Dict[str, Any] = field(default_factory=dict)

def branchpoint(name: str, **metadata) -> BranchPoint:
    return BranchPoint(name=name, metadata=metadata)

def record_score(value: float, context: str = "") -> ScoreSignal:
    return ScoreSignal(value=value, context=context)

def effect(func: Callable, *args, key: Union[str, None] = None, **kwargs) -> Effect:
    return Effect(func=func, args=args, kwargs=kwargs, key=key)

def choose(options: List[Any], name: str = "choice") -> BranchPoint:
    """
    Helper to create a BranchPoint with explicit options.
    """
    return BranchPoint(name=name, metadata={"options": options})
