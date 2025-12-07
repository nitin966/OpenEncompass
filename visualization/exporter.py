import os
import json
from typing import Union, List
from runtime.node import SearchNode

try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

def export_to_dot(source: Union[str, List[SearchNode]], output_filename: str = "search_tree"):
    """
    Generates a GraphViz DOT file from a trace folder or a list of SearchNodes.
    """
    if Digraph is None:
        print("Error: 'graphviz' library not found. Install it via 'pip install graphviz'.")
        return

    dot = Digraph(comment='EnCompass Search Tree')
    dot.attr(rankdir='TB')  # Top to Bottom layout

    # Load all nodes
    nodes_data = []
    
    if isinstance(source, str):
        # Load from folder
        trace_folder = source
        if not os.path.exists(trace_folder):
             print(f"Trace folder {trace_folder} does not exist.")
             return
             
        files = [f for f in os.listdir(trace_folder) if f.endswith('.json')]
        if not files:
            print(f"No trace files found in {trace_folder}")
            return
        print(f"Visualizing {len(files)} nodes from disk...")
        
        for f in files:
            path = os.path.join(trace_folder, f)
            with open(path, 'r') as fd:
                data = json.load(fd)
                # Normalize to dict with expected keys
                nodes_data.append(data)
    else:
        # Use provided nodes
        print(f"Visualizing {len(source)} nodes from memory...")
        for node in source:
            nodes_data.append({
                "id": str(node.node_id),
                "parent": str(node.parent_id) if node.parent_id else None,
                "score": node.score,
                "action": node.action_taken,
                "terminal": node.is_terminal,
                "meta": node.metadata
            })

    # Build Graph
    # Map ID to data for easy parent lookup if needed
    nodes_map = {d['id']: d for d in nodes_data}

    for data in nodes_data:
        nid = data['id']
        # Label format: "Score: 0.5\nAction: 5"
        label = f"Score: {data['score']:.2f}\nInput: {data['action']}"
        if data['terminal']:
            label += f"\nResult: {data.get('meta', {}).get('result')}"
            shape = 'box'
            color = 'lightblue' if data['score'] > 0 else 'salmon' # Blue for success, Red for fail
        else:
            shape = 'ellipse'
            color = 'black'

        dot.node(nid, label=label, shape=shape, color=color)

        # Draw edge from parent
        parent_id = data['parent']
        if parent_id and parent_id in nodes_map:
            dot.edge(parent_id, nid)

    # Render
    output_path = dot.render(output_filename, format='png', view=False)
    print(f"Search tree saved to: {output_path}")
