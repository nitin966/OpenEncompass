import os
import json
try:
    from graphviz import Digraph
except ImportError:
    Digraph = None

def export_to_dot(trace_folder: str, output_filename: str = "search_tree"):
    """
    Scans the trace folder and generates a GraphViz DOT file.
    """
    if Digraph is None:
        print("Error: 'graphviz' library not found. Install it via 'pip install graphviz'.")
        return

    dot = Digraph(comment='EnCompass Search Tree')
    dot.attr(rankdir='TB')  # Top to Bottom layout

    # Load all nodes
    nodes = {}
    files = [f for f in os.listdir(trace_folder) if f.endswith('.json')]
    
    if not files:
        print(f"No trace files found in {trace_folder}")
        return

    print(f"Visualizing {len(files)} nodes...")

    for f in files:
        path = os.path.join(trace_folder, f)
        with open(path, 'r') as fd:
            data = json.load(fd)
            nodes[data['id']] = data

    # Build Graph
    for nid, data in nodes.items():
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
        if parent_id and parent_id in nodes:
            dot.edge(parent_id, nid)

    # Render
    output_path = dot.render(output_filename, format='png', view=False)
    print(f"Search tree saved to: {output_path}")
