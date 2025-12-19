"""ARC-style tasks for hypothesis search experiment.

Each task is a pattern discovery problem:
- Given input/output pairs, find the transformation function.
- The agent must generate a Python function that implements the pattern.
"""

from dataclasses import dataclass
from typing import List, Callable, Any


@dataclass
class HypothesisTask:
    """A single hypothesis search task."""
    name: str
    description: str
    train_inputs: List[Any]
    train_outputs: List[Any]
    test_inputs: List[Any]
    test_outputs: List[Any]
    difficulty: str  # "easy", "medium", "hard"
    
    def validate(self, func: Callable) -> tuple[float, int, int]:
        """Validate a hypothesis function against test cases.
        
        Returns:
            Tuple of (accuracy, correct_count, total_count)
        """
        correct = 0
        total = len(self.test_inputs)
        
        for inp, expected in zip(self.test_inputs, self.test_outputs):
            try:
                result = func(inp)
                if result == expected:
                    correct += 1
            except Exception:
                pass
        
        return correct / total if total > 0 else 0.0, correct, total


# ============================================================================
# TASK COLLECTION
# ============================================================================

TASKS = [
    # Easy: Simple arithmetic
    HypothesisTask(
        name="double",
        description="Multiply by 2",
        train_inputs=[1, 2, 3],
        train_outputs=[2, 4, 6],
        test_inputs=[4, 5, 10, 100],
        test_outputs=[8, 10, 20, 200],
        difficulty="easy",
    ),
    
    HypothesisTask(
        name="add_five",
        description="Add 5 to each number",
        train_inputs=[0, 1, 2],
        train_outputs=[5, 6, 7],
        test_inputs=[3, 10, 100],
        test_outputs=[8, 15, 105],
        difficulty="easy",
    ),
    
    HypothesisTask(
        name="triple",
        description="Multiply by 3",
        train_inputs=[1, 2, 3],
        train_outputs=[3, 6, 9],
        test_inputs=[4, 5, 10],
        test_outputs=[12, 15, 30],
        difficulty="easy",
    ),
    
    # Medium: Slightly more complex patterns
    HypothesisTask(
        name="square",
        description="Square the number",
        train_inputs=[1, 2, 3, 4],
        train_outputs=[1, 4, 9, 16],
        test_inputs=[5, 6, 10],
        test_outputs=[25, 36, 100],
        difficulty="medium",
    ),
    
    HypothesisTask(
        name="double_plus_one",
        description="Multiply by 2 and add 1",
        train_inputs=[0, 1, 2, 3],
        train_outputs=[1, 3, 5, 7],
        test_inputs=[4, 5, 10],
        test_outputs=[9, 11, 21],
        difficulty="medium",
    ),
    
    HypothesisTask(
        name="subtract_from_10",
        description="Subtract from 10",
        train_inputs=[0, 1, 2, 5],
        train_outputs=[10, 9, 8, 5],
        test_inputs=[3, 4, 10],
        test_outputs=[7, 6, 0],
        difficulty="medium",
    ),
    
    # Hard: Complex patterns
    HypothesisTask(
        name="cube",
        description="Cube the number",
        train_inputs=[1, 2, 3],
        train_outputs=[1, 8, 27],
        test_inputs=[4, 5],
        test_outputs=[64, 125],
        difficulty="hard",
    ),
    
    HypothesisTask(
        name="factorial_indicator",
        description="1 if factorial > 100, else 0",
        train_inputs=[1, 2, 3, 4, 5],
        train_outputs=[0, 0, 0, 0, 1],
        test_inputs=[6, 7, 3],
        test_outputs=[1, 1, 0],
        difficulty="hard",
    ),
]


def get_tasks(difficulty: str = None) -> List[HypothesisTask]:
    """Get tasks, optionally filtered by difficulty."""
    if difficulty:
        return [t for t in TASKS if t.difficulty == difficulty]
    return TASKS


def get_task_by_name(name: str) -> HypothesisTask:
    """Get a specific task by name."""
    for task in TASKS:
        if task.name == name:
            return task
    raise ValueError(f"Task '{name}' not found")


if __name__ == "__main__":
    print("Available Hypothesis Search Tasks:")
    print("-" * 50)
    for task in TASKS:
        print(f"\n{task.name} [{task.difficulty}]")
        print(f"  {task.description}")
        print(f"  Train: {task.train_inputs} -> {task.train_outputs}")
        print(f"  Test:  {task.test_inputs} -> {task.test_outputs}")
