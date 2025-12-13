from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4


@dataclass
class SearchNode:
    """
    Represents a node in the search tree, corresponding to a specific execution path.

    Attributes:
        trace_history: List of inputs (choices) made to reach this state.
        score: Cumulative score of this path.
        depth: Number of decisions made (length of history).
        parent_id: UUID of the parent node (None for root).
        node_id: Unique UUID for this node.
        is_terminal: True if the agent execution has completed.
        action_taken: String representation of the last action taken to reach this node.
        metadata: Arbitrary metadata (e.g., final result, debug info).
    """

    trace_history: list[Any]  # The inputs injected so far
    score: float = 0.0
    depth: int = 0
    node_id: UUID = field(default_factory=uuid4)
    parent_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    is_terminal: bool = False
    action_taken: str = "root"
    machine_state: dict[str, Any] | None = None  # Serialized state of the AgentMachine

    def __hash__(self):
        return hash(self.node_id)
