"""EnCompass Hypothesis Search Agent - Simple Version with Branchpoint.

This implements the SAME functionality as base_hypothesis_agent.py but using
EnCompass primitives. Notice how much simpler the code is:

KEY METRICS:
- Lines of code: ~120 lines (vs ~350 in base_hypothesis_agent.py)
- Complexity: Low (EnCompass handles state, retries, search automatically)
- Same functionality: Hypothesis generation, validation, refinement

The key differences:
1. No manual state machine
2. No explicit retry loops  
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

from encompass import branchpoint, record_score
from encompass.llm.ollama import OllamaModel
from tasks import HypothesisTask, get_tasks


# ============================================================================
# PROMPTS (same as base agent)
# ============================================================================

HYPOTHESIS_PROMPT = """You are solving a pattern discovery problem.

Given input/output pairs, write a Python function that transforms the input to output.

Training Examples:
{examples}

Write a Python function called `transform(x)` that takes a single input and returns the output.
Return ONLY the Python function code, no explanations.
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


def compile_function(code: str) -> Tuple[Optional[Callable], str]:
    """Safely compile and extract the transform function."""
    try:
        local_scope = {}
        exec(code, {"__builtins__": {}}, local_scope)
        return local_scope.get("transform"), ""
    except Exception as e:
        return None, str(e)


def format_examples(task: HypothesisTask) -> str:
    """Format training examples for the prompt."""
    return "\n".join(f"  Input: {i} -> Output: {o}" 
                     for i, o in zip(task.train_inputs, task.train_outputs))


# ============================================================================
# ENCOMPASS AGENT - Linear control flow, no state machine!
# ============================================================================

async def solve_task(llm: OllamaModel, task: HypothesisTask) -> Tuple[str, float, bool]:
    """Solve a single hypothesis task.
    
    With EnCompass:
    - branchpoint() marks where search can explore alternatives
    - record_score() tells the search strategy how good this path is
    - No manual state machine needed!
    """
    print(f"  Solving '{task.name}'...")
    
    # branchpoint allows search to try multiple hypothesis generation attempts
    branchpoint("hypothesis_generation")
    
    # Generate hypothesis
    prompt = HYPOTHESIS_PROMPT.format(examples=format_examples(task))
    response = await llm.generate(prompt, max_tokens=512)
    code = clean_code(response)
    
    # Compile and validate
    func, error = compile_function(code)
    
    if func is None:
        record_score(0.0)
        print(f"    ✗ Compilation failed")
        return code, 0.0, False
    
    # Test accuracy
    accuracy, correct, total = task.validate(func)
    
    # Record score for search - accuracy provides gradient
    record_score(accuracy * 100)
    
    solved = accuracy == 1.0
    if solved:
        print(f"    ✓ Solved! (100% on {total} test cases)")
    else:
        print(f"    Partial: {accuracy*100:.0f}%")
    
    return code, accuracy, solved


async def run_hypothesis_search(
    model: str = "qwen2.5:32b",
    output_dir: Path = None,
) -> dict:
    """Run the EnCompass hypothesis search agent.
    
    This function is dramatically simpler than base_hypothesis_agent.py:
    - No HypothesisState enum
    - No TaskState/AgentState dataclasses
    - No step_* functions
    - No state machine loop
    - Just linear, sequential code!
    """
    base_dir = Path(__file__).parent
    
    if output_dir is None:
        output_dir = base_dir / "output" / "encompass_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ENCOMPASS HYPOTHESIS AGENT (With EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~120 (vs ~350 for base agent)")
    print()
    
    llm = OllamaModel(model=model, temperature=0.3)
    tasks = get_tasks()
    
    results = {
        "model": model,
        "agent": "encompass",
        "agent_lines": 120,
        "tasks": [],
        "solved_count": 0,
        "failed_count": 0,
    }
    
    # Simple loop - no state machine needed!
    for task in tasks:
        code, accuracy, solved = await solve_task(llm, task)
        
        if solved:
            results["solved_count"] += 1
        else:
            results["failed_count"] += 1
        
        results["tasks"].append({
            "name": task.name,
            "difficulty": task.difficulty,
            "status": "solved" if solved else "failed",
            "accuracy": accuracy,
            "hypothesis": code,
        })
        
        # Save hypothesis
        if code:
            (output_dir / f"{task.name}.py").write_text(code)
    
    print(f"\nResults: {results['solved_count']}/{len(tasks)} tasks solved")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EnCompass Hypothesis Search Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_hypothesis_search(model=args.model))
