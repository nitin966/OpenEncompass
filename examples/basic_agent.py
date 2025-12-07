"""
Basic agent example - simplest possible use of the CPS compiler.

This demonstrates thecore concepts:
- Compiling a generator function
- Yielding branchpoints to create choices
- Using the execution engine
- Running a search strategy
"""

from encompass import compile, branchpoint
from runtime.engine import ExecutionEngine
from search.strategies import BeamSearch
from storage.filesystem import FileSystemStore
import asyncio


@compile
def simple_agent():
    """
    An agent that makes two choices and returns their sum.
    
    This is the minimal example - no arguments, just the basics.
    """
    # First choice
    x = yield branchpoint("first_choice")
    
    # Second choice  
    y = yield branchpoint("second_choice")
    
    # Return result
    return x + y


# Sampler returns options for each branchpoint
async def sampler(node, metadata=None):
    """Return available choices: [1, 2, 3]"""
    return [1, 2, 3]


async def main():
    # Create engine and search strategy
    engine = ExecutionEngine()
    store = FileSystemStore("./data")
    
    beam = BeamSearch(
        store=store,
        engine=engine,
        sampler=sampler,
        width=3  # Keep top 3 paths
    )
    
    # Run search
    results = await beam.search(simple_agent)
    
    # Print results
    print(f"Found {len(results)} completed paths:")
    for node in results:
        if node.is_terminal:
            print(f"  Result: {node.metadata['result']}, Score: {node.score}")


if __name__ == "__main__":
    asyncio.run(main())
