from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass(frozen=True)
class ControlSignal:
    pass

@dataclass(frozen=True)
class BranchPoint(ControlSignal):
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ScoreSignal(ControlSignal):
    value: float
    context: str = ""

def branchpoint(name: str, **metadata) -> BranchPoint:
    return BranchPoint(name=name, metadata=metadata)

def record_score(value: float, context: str = "") -> ScoreSignal:
    return ScoreSignal(value=value, context=context)
