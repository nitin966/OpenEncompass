from typing import Any, Protocol

from runtime.node import SearchNode


class Sampler(Protocol):
    async def __call__(self, node: SearchNode, metadata: dict[str, Any] = None) -> list[Any]:
        """
        Given a search node, return a list of possible inputs (actions) to take.
        """
        ...
