from abc import ABC, abstractmethod
from uuid import UUID

from runtime.node import SearchNode


class StateStore(ABC):
    """
    Abstract base class for persisting search nodes.
    """

    @abstractmethod
    def save_node(self, node: SearchNode) -> None:
        """Saves a single search node to storage."""
        pass

    @abstractmethod
    def load_node(self, node_id: UUID) -> SearchNode | None:
        """Loads a search node by its ID."""
        pass
