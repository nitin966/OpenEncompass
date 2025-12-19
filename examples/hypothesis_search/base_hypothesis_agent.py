"""Base Hypothesis Search Agent (WITHOUT EnCompass) - State Machine Version.

This implements the hypothesis search with explicit state management,
retry loops, and validation - all the complexity that EnCompass abstracts away.

KEY METRICS:
- Lines of code: ~350 lines
- Complexity: High (manual state, retries, validation loops)
- Compare to: encompass_hypothesis_agent.py (~100 lines with same functionality)
"""

import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from encompass.llm.ollama import OllamaModel
from tasks import HypothesisTask, get_tasks


# ============================================================================
# STATE DEFINITIONS - Manual state management required without EnCompass
# ============================================================================

class HypothesisState(Enum):
    """States in the hypothesis search state machine."""
    INIT = "init"
    GENERATING = "generating"
    VALIDATING = "validating"
    REFINING = "refining"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """State for a single task - manual bookkeeping."""
    task: HypothesisTask
    current_hypothesis: str = ""
    current_func: Optional[Callable] = None
    state: HypothesisState = HypothesisState.INIT
    attempts: int = 0
    max_attempts: int = 3
    accuracy: float = 0.0
    best_hypothesis: str = ""
    best_accuracy: float = 0.0
    error_message: str = ""


@dataclass
class AgentState:
    """Global agent state - must track all variables explicitly."""
    model_name: str
    task_states: Dict[str, TaskState] = field(default_factory=dict)
    current_task_idx: int = 0
    total_tasks: int = 0
    solved_count: int = 0
    failed_count: int = 0
    total_attempts: int = 0


# ============================================================================
# PROMPTS
# ============================================================================

HYPOTHESIS_PROMPT = """You are solving a pattern discovery problem.

Given input/output pairs, write a Python function that transforms the input to output.

Training Examples:
{examples}

Write a Python function called `transform(x)` that takes a single input and returns the output.

Requirements:
1. Function must be named exactly `transform`
2. Function takes a single argument `x`
3. Return the transformed value

Return ONLY the Python function code, no explanations.
"""

REFINE_PROMPT = """Your previous function was incorrect.

Previous function:
```python
{previous_code}
```

Expected results:
{expected}

Actual results:
{actual}

Write a corrected Python function `transform(x)` that passes ALL test cases.
Return ONLY the corrected Python code.
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
        func = local_scope.get("transform")
        if func is None:
            return None, "No 'transform' function found"
        return func, ""
    except SyntaxError as e:
        return None, f"Syntax error: {e}"
    except Exception as e:
        return None, f"Execution error: {e}"


def format_examples(task: HypothesisTask) -> str:
    """Format training examples for the prompt."""
    lines = []
    for inp, out in zip(task.train_inputs, task.train_outputs):
        lines.append(f"  Input: {inp} -> Output: {out}")
    return "\n".join(lines)


def format_results(task: HypothesisTask, func: Callable) -> Tuple[str, str]:
    """Format expected vs actual results for refinement."""
    expected_lines = []
    actual_lines = []
    
    for inp, expected_out in zip(task.train_inputs, task.train_outputs):
        expected_lines.append(f"  transform({inp}) = {expected_out}")
        try:
            actual = func(inp)
            actual_lines.append(f"  transform({inp}) = {actual}")
        except Exception as e:
            actual_lines.append(f"  transform({inp}) raised {type(e).__name__}")
    
    return "\n".join(expected_lines), "\n".join(actual_lines)


# ============================================================================
# STATE MACHINE STEP FUNCTIONS - The complexity EnCompass eliminates
# ============================================================================

async def step_init(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, HypothesisState]:
    """Initialize task states - state machine entry point."""
    for task in get_tasks():
        task_state = TaskState(task=task)
        state.task_states[task.name] = task_state
        state.total_tasks += 1
    
    if state.total_tasks == 0:
        return state, HypothesisState.FAILED
    
    return state, HypothesisState.GENERATING


async def step_generate(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, HypothesisState]:
    """Generate hypothesis for current task."""
    task_names = list(state.task_states.keys())
    if state.current_task_idx >= len(task_names):
        return state, HypothesisState.COMPLETED
    
    current_name = task_names[state.current_task_idx]
    task_state = state.task_states[current_name]
    
    # Skip if already solved
    if task_state.state == HypothesisState.COMPLETED:
        state.current_task_idx += 1
        return state, HypothesisState.GENERATING
    
    # Check attempt limit
    if task_state.attempts >= task_state.max_attempts:
        task_state.state = HypothesisState.FAILED
        state.failed_count += 1
        # Use best so far
        if task_state.best_hypothesis:
            task_state.current_hypothesis = task_state.best_hypothesis
        state.current_task_idx += 1
        return state, HypothesisState.GENERATING
    
    task_state.attempts += 1
    state.total_attempts += 1
    print(f"  Generating hypothesis for '{task_state.task.name}' (attempt {task_state.attempts}/{task_state.max_attempts})...")
    
    try:
        prompt = HYPOTHESIS_PROMPT.format(examples=format_examples(task_state.task))
        response = await llm.generate(prompt, max_tokens=512)
        code = clean_code(response)
        task_state.current_hypothesis = code
        task_state.state = HypothesisState.VALIDATING
        
        return state, HypothesisState.VALIDATING
        
    except Exception as e:
        task_state.error_message = str(e)
        print(f"    Error: {e}")
        return state, HypothesisState.GENERATING


async def step_validate(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, HypothesisState]:
    """Validate current hypothesis."""
    task_names = list(state.task_states.keys())
    current_name = task_names[state.current_task_idx]
    task_state = state.task_states[current_name]
    
    # Compile the function
    func, error = compile_function(task_state.current_hypothesis)
    
    if func is None:
        print(f"    Compilation failed: {error}")
        task_state.error_message = error
        task_state.state = HypothesisState.REFINING
        return state, HypothesisState.REFINING
    
    task_state.current_func = func
    
    # Validate on training data first
    train_accuracy, correct, total = validate_on_examples(
        func, task_state.task.train_inputs, task_state.task.train_outputs
    )
    
    if train_accuracy == 1.0:
        # Test on held-out data
        test_accuracy, correct, total = task_state.task.validate(func)
        task_state.accuracy = test_accuracy
        
        if test_accuracy == 1.0:
            print(f"    ✓ Solved! (100% accuracy on {total} test cases)")
            task_state.state = HypothesisState.COMPLETED
            state.solved_count += 1
            state.current_task_idx += 1
            return state, HypothesisState.GENERATING
        else:
            print(f"    Partial: {test_accuracy*100:.0f}% on test set")
            # Track best so far
            if test_accuracy > task_state.best_accuracy:
                task_state.best_accuracy = test_accuracy
                task_state.best_hypothesis = task_state.current_hypothesis
    else:
        print(f"    Training accuracy: {train_accuracy*100:.0f}%")
    
    # Need refinement
    task_state.state = HypothesisState.REFINING
    return state, HypothesisState.REFINING


async def step_refine(state: AgentState, llm: OllamaModel) -> Tuple[AgentState, HypothesisState]:
    """Refine hypothesis based on errors."""
    task_names = list(state.task_states.keys())
    current_name = task_names[state.current_task_idx]
    task_state = state.task_states[current_name]
    
    # Check if we should continue
    if task_state.attempts >= task_state.max_attempts:
        task_state.state = HypothesisState.FAILED
        state.failed_count += 1
        if task_state.best_hypothesis:
            task_state.current_hypothesis = task_state.best_hypothesis
        state.current_task_idx += 1
        return state, HypothesisState.GENERATING
    
    task_state.attempts += 1
    state.total_attempts += 1
    print(f"  Refining hypothesis (attempt {task_state.attempts}/{task_state.max_attempts})...")
    
    try:
        if task_state.current_func:
            expected, actual = format_results(task_state.task, task_state.current_func)
        else:
            expected = format_examples(task_state.task)
            actual = f"Code failed to compile: {task_state.error_message}"
        
        prompt = REFINE_PROMPT.format(
            previous_code=task_state.current_hypothesis,
            expected=expected,
            actual=actual
        )
        response = await llm.generate(prompt, max_tokens=512)
        code = clean_code(response)
        task_state.current_hypothesis = code
        task_state.state = HypothesisState.VALIDATING
        
        return state, HypothesisState.VALIDATING
        
    except Exception as e:
        task_state.error_message = str(e)
        return state, HypothesisState.GENERATING


def validate_on_examples(func: Callable, inputs: List, outputs: List) -> Tuple[float, int, int]:
    """Validate function on examples."""
    correct = 0
    total = len(inputs)
    
    for inp, expected in zip(inputs, outputs):
        try:
            if func(inp) == expected:
                correct += 1
        except Exception:
            pass
    
    return correct / total if total > 0 else 0.0, correct, total


# ============================================================================
# MAIN STATE MACHINE DRIVER
# ============================================================================

async def run_state_machine(state: AgentState, llm: OllamaModel) -> AgentState:
    """Run the hypothesis search state machine."""
    current_state = HypothesisState.INIT
    
    while current_state not in (HypothesisState.COMPLETED, HypothesisState.FAILED):
        if current_state == HypothesisState.INIT:
            state, current_state = await step_init(state, llm)
            
        elif current_state == HypothesisState.GENERATING:
            state, current_state = await step_generate(state, llm)
            if state.current_task_idx >= state.total_tasks:
                current_state = HypothesisState.COMPLETED
                
        elif current_state == HypothesisState.VALIDATING:
            state, current_state = await step_validate(state, llm)
            
        elif current_state == HypothesisState.REFINING:
            state, current_state = await step_refine(state, llm)
    
    return state


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def run_hypothesis_search(
    model: str = "qwen2.5:32b",
    output_dir: Path = None,
) -> dict:
    """Run the base hypothesis search agent."""
    base_dir = Path(__file__).parent
    
    if output_dir is None:
        output_dir = base_dir / "output" / "base_agent"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("BASE HYPOTHESIS AGENT (Without EnCompass)")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Output: {output_dir}")
    print(f"Lines of code in this agent: ~350")
    print()
    
    llm = OllamaModel(model=model, temperature=0.3)
    
    state = AgentState(model_name=model)
    final_state = await run_state_machine(state, llm)
    
    # Build results
    results = {
        "model": model,
        "agent": "base",
        "agent_lines": 350,
        "tasks": [],
        "solved_count": final_state.solved_count,
        "failed_count": final_state.failed_count,
        "total_attempts": final_state.total_attempts,
    }
    
    for name, task_state in final_state.task_states.items():
        results["tasks"].append({
            "name": name,
            "difficulty": task_state.task.difficulty,
            "status": "solved" if task_state.state == HypothesisState.COMPLETED else "failed",
            "accuracy": task_state.accuracy if task_state.state == HypothesisState.COMPLETED else task_state.best_accuracy,
            "attempts": task_state.attempts,
            "hypothesis": task_state.current_hypothesis or task_state.best_hypothesis,
        })
        
        # Save hypothesis to file
        if task_state.current_hypothesis or task_state.best_hypothesis:
            code = task_state.current_hypothesis or task_state.best_hypothesis
            (output_dir / f"{name}.py").write_text(code)
    
    print(f"\nResults: {results['solved_count']}/{final_state.total_tasks} tasks solved")
    print(f"Total LLM calls: {results['total_attempts']}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Base Hypothesis Search Agent")
    parser.add_argument("--model", default="qwen2.5:32b", help="Ollama model name")
    args = parser.parse_args()
    
    results = asyncio.run(run_hypothesis_search(model=args.model))
