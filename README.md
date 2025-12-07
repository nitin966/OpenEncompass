# EnCompass: Search-Based Agent Framework

EnCompass is a Python framework for building "Program-in-Control" AI agents. It separates the **workflow logic** (the agent's code) from the **search strategy** (how the agent explores possibilities), allowing for powerful, non-deterministic reasoning without cluttering the business logic.

This implementation achieves parity with the system described in the [Asari AI blog post](https://asari.ai/blog/encompass).

## Features

- **Search Strategies**: Includes **Beam Search** and **Monte Carlo Tree Search (MCTS)** with UCT selection and random rollouts.
- **Clean API**: Use `@compile`, `branchpoint`, and `record_score` to define agents as standard Python generators.
- **LLM Abstraction**: Built-in `LanguageModel` protocol to easily swap between Mock LLMs (for testing) and real providers (e.g., OpenAI).
- **Visualization**: Export search trees to GraphViz for analysis.
- **Robust Storage**: File-system based storage for search traces.

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install dill graphviz
   ```
3. (Optional) Install `graphviz` system binary if you want visualization.

## Quick Start

### 1. Define an Agent

```python
from encompass import compile, branchpoint, record_score

@compile
def my_agent():
    # Make a choice
    choice = yield branchpoint("pick_option")
    
    if choice == 1:
        yield record_score(10.0)
        return "Success"
    else:
        yield record_score(0.0)
        return "Failure"
```

### 2. Run with Search

```python
from runtime.engine import ExecutionEngine
from storage.filesystem import FileSystemStore
from search.strategies import BeamSearch

# Define a sampler (returns possible inputs for a node)
def sampler(node):
    return [0, 1]

searcher = BeamSearch(
    store=FileSystemStore(),
    engine=ExecutionEngine(),
    sampler=sampler,
    width=2
)

results = searcher.search(my_agent)
print("Top Result:", results[0].metadata['result'])
```

## Examples

### Code Translation Agent

A realistic example simulating translating Python code to C++ using an LLM.

```bash
python3 run_translation.py
```

This script:
1. Runs the agent with **Beam Search**.
2. Runs the agent with **MCTS**.
3. Generates a visualization of the search tree (`translation_search_tree.png`).

## Testing

Run the comprehensive test suite:

```bash
python3 -m unittest discover tests
```

## Project Structure

- `encompass/`: Public API.
- `core/`: Core abstractions (Signals, Decorators, LLM).
- `runtime/`: Execution engine (Replay mechanism).
- `search/`: Search strategies (Beam, MCTS).
- `storage/`: State persistence.
- `examples/`: Example agents.
- `tests/`: Unit and End-to-End tests.
