"""Test cases for hypothesis search experiment."""

import sys
from pathlib import Path

# Add parent directory for imports
base_dir = Path(__file__).parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from tasks import TASKS, HypothesisTask


def test_generated_function(task_name: str, code: str) -> tuple[bool, float, str]:
    """Test a generated function against expected outputs.
    
    Args:
        task_name: Name of the task
        code: Python code containing the transform function
        
    Returns:
        Tuple of (passed, accuracy, message)
    """
    # Find the task
    task = None
    for t in TASKS:
        if t.name == task_name:
            task = t
            break
    
    if task is None:
        return False, 0.0, f"Unknown task: {task_name}"
    
    # Compile the function
    try:
        local_scope = {}
        exec(code, {"__builtins__": {}}, local_scope)
        func = local_scope.get("transform")
        if func is None:
            return False, 0.0, "No 'transform' function found"
    except Exception as e:
        return False, 0.0, f"Compilation error: {e}"
    
    # Validate
    accuracy, correct, total = task.validate(func)
    passed = accuracy == 1.0
    
    return passed, accuracy, f"{correct}/{total} correct"


def run_tests(output_dir: str) -> tuple[int, int, list]:
    """Run all tests on generated functions.
    
    Args:
        output_dir: Directory containing generated Python files
        
    Returns:
        Tuple of (passed, failed, errors)
    """
    output_path = Path(output_dir)
    passed = 0
    failed = 0
    errors = []
    
    if not output_path.exists():
        return 0, len(TASKS), [f"Output directory not found: {output_dir}"]
    
    for task in TASKS:
        file_path = output_path / f"{task.name}.py"
        
        if not file_path.exists():
            failed += 1
            errors.append(f"{task.name}: File not found")
            continue
        
        code = file_path.read_text()
        success, accuracy, message = test_generated_function(task.name, code)
        
        if success:
            passed += 1
        else:
            failed += 1
            errors.append(f"{task.name}: {message} (accuracy: {accuracy*100:.0f}%)")
    
    return passed, failed, errors


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", help="Directory with generated functions")
    args = parser.parse_args()
    
    passed, failed, errors = run_tests(args.output_dir)
    
    print(f"\nResults: {passed} passed, {failed} failed")
    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
