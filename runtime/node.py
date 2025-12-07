from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from uuid import UUID, uuid4

@dataclass
class SearchNode:
    """
    Stores the SEQUENCE of choices made to reach this state.
    We re-play this sequence to reconstruct the state.
    """
    trace_history: List[Any]  # The inputs injected so far
    score: float = 0.0
    depth: int = 0
    node_id: UUID = field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    is_terminal: bool = False
    action_taken: str = "root"

    def __hash__(self):
        return hash(self.node_id)
