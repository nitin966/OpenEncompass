import os
import shutil
import random
from examples.translation_agent import create_translation_agent
from core.llm import MockLLM
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch, MCTS
from visualization.exporter import export_to_dot

# Sampler for the translation agent
def translation_sampler(node):
    # Signature style (3 options)
    if node.depth == 0:
        return [0, 1, 2]
    # Body style (2 options)
    elif node.depth == 1:
        return [0, 1]
    return []

if __name__ == "__main__":
    trace_dir = "encompass_trace"
    if os.path.exists(trace_dir): shutil.rmtree(trace_dir)
    
    print("--- Running EnCompass Translation Demo ---")
    
    # Initialize LLM
    llm = MockLLM()
    agent = create_translation_agent(llm)
    
    # Beam Search
    print("\n[Beam Search]")
    beam = BeamSearch(
        store=FileSystemStore(),
        engine=ExecutionEngine(),
        sampler=translation_sampler,
        width=2,
        max_depth=5
    )
    results_beam = beam.search(agent)
    results_beam.sort(key=lambda n: n.score, reverse=True)
    
    if results_beam:
        print(f"Top Result (Score {results_beam[0].score}):")
        print(results_beam[0].metadata.get('result'))
    
    # MCTS
    print("\n[MCTS]")
    mcts = MCTS(
        store=FileSystemStore(),
        engine=ExecutionEngine(),
        sampler=translation_sampler,
        iterations=20, 
        exploration_weight=10.0
    )
    results_mcts = mcts.search(agent)
    results_mcts.sort(key=lambda n: n.score, reverse=True)
    
    if results_mcts:
        print(f"Top Result (Score {results_mcts[0].score}):")
        print(results_mcts[0].metadata.get('result'))

    # Visualize
    print("\nGenerating Visualization...")
    export_to_dot(trace_dir, "translation_search_tree")
