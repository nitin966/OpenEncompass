import json
import pickle
from pathlib import Path
from typing import Any
from uuid import UUID

from runtime.node import SearchNode
from storage.base import StateStore


class FileSystemStore(StateStore):
    """
    File-system based StateStore with cache persistence.
    Saves nodes as JSON and execution cache as pickle for resumable searches.
    """

    def __init__(self, base_path: str = "encompass_trace"):
        """
        Args:
            base_path: The directory where trace files will be stored.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.base_path / "execution_cache.pkl"

    def save_node(self, node: SearchNode) -> None:
        """Save node to filesystem."""
        meta_path = self.base_path / f"{node.node_id}.json"

        data = {
            "id": str(node.node_id),
            "parent": str(node.parent_id) if node.parent_id else None,
            "score": node.score,
            "depth": node.depth,
            "action": node.action_taken,
            "terminal": node.is_terminal,
            "history": node.trace_history,
            "meta": node.metadata,
        }
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=2)

    def load_node(self, node_id: UUID) -> SearchNode:
        """Load node from filesystem."""
        meta_path = self.base_path / f"{node_id}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Node {node_id} not found")

        with open(meta_path) as f:
            data = json.load(f)

        return SearchNode(
            trace_history=data["history"],
            score=data["score"],
            depth=data["depth"],
            node_id=UUID(data["id"]),
            parent_id=UUID(data["parent"]) if data["parent"] else None,
            metadata=data.get("meta", {}),
            is_terminal=data.get("terminal", False),
            action_taken=data.get("action", "unknown"),
        )

    def save_cache(self, cache_dict: dict[str, Any]) -> None:
        """
        Save execution cache to disk for resumable searches.

        Args:
            cache_dict: The engine's execution cache
        """
        with open(self.cache_file, "wb") as f:
            pickle.dump(cache_dict, f)

    def load_cache(self) -> dict[str, Any]:
        """
        Load execution cache from disk.

        Returns:
            Cache dict or empty dict if no cache exists
        """
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, "rb") as f:
                from typing import cast

                return cast(dict[str, Any], pickle.load(f))
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return {}

    def clear_cache(self) -> None:
        """Clear the execution cache file."""
        if self.cache_file.exists():
            self.cache_file.unlink()
