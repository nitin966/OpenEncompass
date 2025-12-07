import os
import json
from uuid import UUID
from storage.base import StateStore
from runtime.node import SearchNode

class FileSystemStore(StateStore):
    def __init__(self, base_path="encompass_trace"):
        self.base_path = base_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def save_node(self, node: SearchNode) -> None:
        meta_path = os.path.join(self.base_path, f"{node.node_id}.json")
        
        data = {
            "id": str(node.node_id),
            "parent": str(node.parent_id) if node.parent_id else None,
            "score": node.score,
            "depth": node.depth,
            "action": node.action_taken,
            "terminal": node.is_terminal,
            "history": node.trace_history, # Simple list serialization
            "meta": node.metadata
        }
        with open(meta_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_node(self, node_id: UUID) -> SearchNode:
        meta_path = os.path.join(self.base_path, f"{node_id}.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Node {node_id} not found")
            
        with open(meta_path, 'r') as f:
            data = json.load(f)

        return SearchNode(
            trace_history=data['history'],
            score=data['score'],
            depth=data['depth'],
            node_id=UUID(data['id']),
            parent_id=UUID(data['parent']) if data['parent'] else None,
            metadata=data.get('meta', {}),
            is_terminal=data.get('terminal', False),
            action_taken=data.get('action', "unknown")
        )
