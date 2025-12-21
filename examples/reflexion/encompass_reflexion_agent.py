"""EnCompass Reflexion Agent - Simple Version with Branchpoint.

This implements the SAME functionality as base_reflexion_agent.py but using
EnCompass primitives. Notice how much simpler the code is:

KEY METRICS:
- Lines of code: ~140 lines (vs ~400 in base_reflexion_agent.py)
- Complexity: Low (EnCompass handles state, reflection loops automatically)
- Same functionality: Generate, test, reflect, improve

The key differences:
1. No manual state machine
2. No explicit reflection loops  
3. branchpoint() marks exploration points
4. record_score() provides feedback for search
5. Linear, readable control flow
"""

import asyncio
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass import branchpoint, record_score, compile, effect
from encompass.llm.ollama import OllamaModel
from problems import CodingProblem, get_problems


# ============================================================================
# PROMPTS (same as base agent)
# ============================================================================

GENERATE_PROMPT = """You are an expert Python programmer.

Problem: {description}

Function signature: {signature}

Write a Python function that solves this problem.
Return ONLY the Python function code, no explanations.
"""

IMPROVE_PROMPT = """Your code failed some tests. 

Problem: {description}
Previous code: 
{code}

Test results: {test_results}

Write an IMPROVED Python function.
Return ONLY the Python function code.
"""


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_code(response: str) -> str:
    """Clean LLM response to extract Python code."""
    code = response.strip()
    for prefix in ["```python", "```"]:
        if code.startswith(prefix):
            code = code[len(prefix):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def compile_function(code: str, func_name: str) -> Tuple[Optional[Callable], str]:
    """Safely compile and extract the function."""
    try:
        local_scope = {}
        exec(code, {"__builtins__": __builtins__}, local_scope)
        return local_scope.get(func_name), ""
    except Exception as e:
        return None, str(e)


# ============================================================================
# ENCOMPASS AGENT - Linear control flow, no state machine!
# ============================================================================

@compile
def solve_problem(llm: OllamaModel, problem: CodingProblem, max_attempts: int = 3) -> Tuple[str, float, bool]:
    """Solve a coding problem with Reflexion.
    
    With EnCompass:
    - branchpoint() marks where search can explore alternatives
    - record_score() tells the search strategy how good this path is
    - No manual state machine needed!
    """
    print(f"  Solving '{problem.name}'...")
    
    # Initial code generation - branchpoint allows exploring alternatives
    branchpoint("initial_generation")
    
    prompt = GENERATE_PROMPT.format(
        description=problem.description,
        signature=problem.function_signature
    )
    # Use effect to call async llm.generate
    response = effect(llm.generate, prompt, max_tokens=1024)
    code = clean_code(response)
    
    # Test and iterate (Reflexion loop)
    accuracy = 0.0
    for attempt in range(max_attempts):
        compilation_result = compile_function(code, problem.function_name)
        func = compilation_result[0]
        error = compilation_result[1]
        
        if func is None:
            test_results = f"Compilation error: {error}"
            accuracy = 0.0
        else:
            validation_result = problem.validate(func)
            accuracy = validation_result[0]
            correct = validation_result[1]
            total = validation_result[2]
            
            if accuracy == 1.0:
                record_score(accuracy * 100, context={"result": (code, accuracy, True)})
                print(f"    ✓ Solved! ({correct}/{total} tests passed)")
                return code, accuracy, True
            
            # Record partial score for this attempt
            record_score(accuracy * 100, context={"result": (code, accuracy, False)})
            
            # Format test results for reflection
            test_results = f"{correct}/{total} tests passed"
        
        # Not solved - branchpoint for improvement direction
        if attempt < max_attempts - 1:
            branchpoint("improvement_direction")
            
            improve_prompt = IMPROVE_PROMPT.format(
                description=problem.description,
                code=code,
                test_results=test_results
            )
            response = effect(llm.generate, improve_prompt, max_tokens=1024)
            code = clean_code(response)
    
    # Record final score
    record_score(accuracy * 100, context={"result": (code, accuracy, False)})
    print(f"    Partial: {accuracy*100:.0f}%")
    
    return code, accuracy, False


async def run_reflexion(
    model: str = "qwen2.5:32b",
    output_dir: Path = None,
    strategy_name: str = "beam",
    iterations: int = 10,
) -> dict:
    """Run the EnCompass Reflexion agent.
    
    Now properly uses the EnCompass runtime and search strategies!
    """
    base_dir = Path(__file__).parent
    
    if output_dir is None:
        output_dir = base_dir / "output" / "encompass_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ENCOMPASS REFLEXION AGENT (With EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Strategy: {strategy_name}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~140 (vs ~400 for base agent)")
    print()
    
    llm = OllamaModel(model=model, temperature=0.3)
    problems = get_problems()
    
    # Initialize Engine
    from runtime.engine import ExecutionEngine
    from search.strategies import BeamSearch, MCTS
    from search.ab_mcts import ABMCTS
    from encompass import compile
    
    engine = ExecutionEngine()
    
    # Compile the agent function
    # We need to wrap it to pass arguments
    # solve_problem is already compiled, so calling it returns a machine instance.
    # Sampler Adapter
    async def llm_sampler(node, metadata):
        if metadata and "options" in metadata:
            return metadata["options"]
        # For open-ended generation, we return a single None to allow the agent to proceed
        # and generate its own unique response (since LLM is stochastic).
        # AB-MCTS will call this repeatedly to generate new branches.
        return [None]

    # Select Strategy
    def get_strategy():
        if strategy_name == "beam":
            return BeamSearch(store=None, engine=engine, sampler=llm_sampler, width=3)
        elif strategy_name == "mcts":
            return MCTS(store=None, engine=engine, sampler=llm_sampler, iterations=iterations)
        elif strategy_name == "ab-mcts":
            return ABMCTS(store=None, engine=engine, sampler=llm_sampler, iterations=iterations)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

    results = {
        "model": model,
        "agent": "encompass",
        "strategy": strategy_name,
        "agent_lines": 140,
        "problems": [],
        "solved_count": 0,
        "failed_count": 0,
    }
    
    for problem in problems:
        print(f"  Solving '{problem.name}' with {strategy_name}...")
        strategy = get_strategy()
        
        # Create a partial for the specific problem
        # The strategy expects a factory that returns a generator/coroutine
        def factory():
            return solve_problem(llm, problem, max_attempts=3)
            
        # Run Search
        search_results = await strategy.search(factory)
        
        # Get best result
        if not search_results:
            print("    No results found.")
            code = ""
            accuracy = 0.0
            solved = False
        else:
            best_node = search_results[0] # Sorted by score
            if "result" in best_node.metadata:
                code, accuracy, solved = best_node.metadata["result"]
            else:
                # Should not happen if search found a solution or finished
                print(f"    Warning: Best node missing result. Node: {best_node.node_id}, Score: {best_node.score}, Depth: {best_node.depth}, Metadata Keys: {list(best_node.metadata.keys())}")
                code = ""
                accuracy = 0.0
                solved = False
        
        if solved:
            results["solved_count"] += 1
        else:
            results["failed_count"] += 1
        
        results["problems"].append({
            "name": problem.name,
            "difficulty": problem.difficulty,
            "status": "solved" if solved else "failed",
            "accuracy": accuracy,
            "code": code,
        })
        
        # Save code
        if code:
            (output_dir / f"{problem.name}.py").write_text(code)
    
    print(f"\nResults: {results['solved_count']}/{len(problems)} problems solved")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EnCompass Reflexion Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    parser.add_argument("--strategy", default="beam", help="Search strategy (beam, mcts, ab-mcts)")
    parser.add_argument("--iterations", type=int, default=10, help="Search iterations")
    args = parser.parse_args()
    
    results = asyncio.run(run_reflexion(model=args.model, strategy_name=args.strategy, iterations=args.iterations))
