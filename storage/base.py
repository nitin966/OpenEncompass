from abc import ABC, abstractmethod
from uuid import UUID
from runtime.node import SearchNode

class StateStore(ABC):
    @abstractmethod
    def save_node(self, node: SearchNode) -> None:
        pass

    @abstractmethod
    def load_node(self, node_id: UUID) -> SearchNode:
        pass
