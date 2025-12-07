"""
Agent with arguments - how to pass parameters to compiled agents.

The CPS compiler doesn't capture closure variables, so you need to pass
them as arguments. This example shows the correct pattern.
"""

from encompass import compile, branchpoint
from runtime.engine import ExecutionEngine
from search.strategies import BestOfNSearch
from storage.filesystem import FileSystemStore
import asyncio


@compile
def math_solver(problem_text, target_answer):
    """
    Solve a math problem by trying different approaches.
    
    Args:
        problem_text: The problem to solve
        target_answer: Expected answer (for scoring)
    """
    # First approach
    approach = yield branchpoint("choose_approach", problem=problem_text)
    
    # Calculate based on approach
    if approach == "add":
        result = 2 + 2
    elif approach == "multiply":
        result = 2 * 2
    else:
        result = 0
    
    # Score based on correctness
    if result == target_answer:
        from core.signals import record_score
        yield record_score(100)  # Good score if correct
    
    return result


def create_solver(problem, answer):
    """
    Factory function that creates an agent with captured arguments.
    
    This is the pattern to use:
    1. Define agent with arguments
    2. Return lambda that calls it with the captured values
    """
    return lambda: math_solver(problem, answer)


async def sampler(node, metadata=None):
    """Sampler that provides different approaches"""
    return ["add", "multiply", "subtract"]


async def main():
    # Create a solver for a specific problem
    problem = "What is 2 + 2?"
    agent = create_solver(problem, 4)
    
    # Run best-of-N search
    engine = ExecutionEngine()
    store = FileSystemStore("./data")
    
    best_of_n = BestOfNSearch(
        store=store,
        engine=engine,
        sampler=sampler,
        n=3  # Try 3 approaches
    )
    
    results = await best_of_n.search(agent)
    
    # Find best result
    best = max(results, key=lambda n: n.score)
    print(f"Best result: {best.metadata['result']} (score: {best.score})")


if __name__ == "__main__":
    asyncio.run(main())
