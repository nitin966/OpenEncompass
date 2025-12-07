from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass(frozen=True)
class ControlSignal:
    """Base class for all control signals yielded by agents."""
    pass

@dataclass(frozen=True)
class BranchPoint(ControlSignal):
    """
    Signal indicating a point where the agent needs to make a decision.
    
    Attributes:
        name: A unique identifier for this branch point.
        metadata: Optional dictionary containing context, options, or schema for the decision.
    """
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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

def branchpoint(name: str, **metadata) -> BranchPoint:
    return BranchPoint(name=name, metadata=metadata)

def record_score(value: float, context: str = "") -> ScoreSignal:
    return ScoreSignal(value=value, context=context)
