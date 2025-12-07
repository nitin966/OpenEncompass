from typing import Callable, List, Any
from runtime.engine import ExecutionEngine
from storage.base import StateStore
from storage.filesystem import FileSystemStore
from runtime.node import SearchNode

class SearchSpace:
    """
    Represents the search space of an EnCompass program.
    It encapsulates the execution engine, storage, and the agent program itself.
    """
    def __init__(self, agent_factory: Callable, engine: ExecutionEngine = None, store: StateStore = None):
        self.agent_factory = agent_factory
        self.engine = engine or ExecutionEngine()
        self.store = store or FileSystemStore()

    async def step(self, node: SearchNode, input_value: Any = None):
        """
        Executes one step in the search space.
        """
        child, signal = await self.engine.step(self.agent_factory, node, input_value)
        self.store.save_node(child)
        return child, signal

    def create_root(self) -> SearchNode:
        return self.engine.create_root()
