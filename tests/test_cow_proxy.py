"""
Tests for Copy-on-Write mutable state isolation.

These tests verify that modifications to lists/dicts in one branch
do not affect other branches.

Note: The CPS compiler uses IMPLICIT yields. Use `x = branchpoint(...)` 
not `x = yield branchpoint(...)`. The call to any ControlSignal-returning
function automatically yields.
"""

import pytest

from core.compiler import compile_agent
from core.signals import branchpoint


def run_to_completion(m, first_input=None):
    """Run machine until done, returning the unwrapped result."""
    sig = m.run(first_input)
    while not m._done:
        sig = m.run(None)
    result = m._result
    # Unwrap CoWList/CoWDict if needed
    if hasattr(result, 'unwrap'):
        result = result.unwrap()
    return result


def test_cow_list_isolation():
    """Test that list mutations in one branch don't affect other branches."""
    
    @compile_agent
    def agent_with_list():
        history = [1, 2, 3]
        choice = branchpoint("choice")  # Implicit yield
        if choice == "A":
            history.append(100)  # Branch A appends 100
        else:
            history.append(200)  # Branch B appends 200
        return history
    
    # Create machine and run to branchpoint
    machine_a = agent_with_list()
    sig = machine_a.run(None)
    assert sig.name == "choice"
    
    # Snapshot for branch B (creates a separate machine)
    machine_b = machine_a.snapshot()
    
    # Branch A: append 100
    result_a = run_to_completion(machine_a, "A")
    
    # Branch B: append 200 (using separate machine)
    result_b = run_to_completion(machine_b, "B")
    
    # Verify isolation: each branch should have its own list
    assert result_a == [1, 2, 3, 100], f"Branch A should have [1,2,3,100], got {result_a}"
    assert result_b == [1, 2, 3, 200], f"Branch B should have [1,2,3,200], got {result_b}"


def test_cow_dict_isolation():
    """Test that dict mutations in one branch don't affect other branches."""
    
    @compile_agent
    def agent_with_dict():
        data = {"x": 1}
        choice = branchpoint("choice")  # Implicit yield
        if choice == "A":
            data["y"] = "from_A"
        else:
            data["y"] = "from_B"
        return data
    
    machine_a = agent_with_dict()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    # Branch A
    result_a = run_to_completion(machine_a, "A")
    
    # Branch B
    result_b = run_to_completion(machine_b, "B")

    # Unwrap if needed
    if hasattr(result_a, 'unwrap'):
        result_a = result_a.unwrap()
    if hasattr(result_b, 'unwrap'):
        result_b = result_b.unwrap()
    
    assert result_a == {"x": 1, "y": "from_A"}
    assert result_b == {"x": 1, "y": "from_B"}


def test_cow_nested_list():
    """Test CoW with nested mutable structures."""
    
    @compile_agent
    def agent_nested():
        items = []
        choice = branchpoint("choice")
        items.append({"value": choice})
        return items
    
    machine_a = agent_nested()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, "first")
    result_b = run_to_completion(machine_b, "second")
    
    # Unwrap
    if hasattr(result_a, 'unwrap'):
        result_a = result_a.unwrap()
    if hasattr(result_b, 'unwrap'):
        result_b = result_b.unwrap()
    
    assert result_a == [{"value": "first"}]
    assert result_b == [{"value": "second"}]


def test_cow_list_multiple_mutations():
    """Test multiple mutations to same list."""
    
    @compile_agent
    def agent_multi_mutate():
        nums = []
        nums.append(1)
        nums.append(2)
        choice = branchpoint("choice")
        nums.append(choice)
        nums.extend([10, 20])
        return nums
    
    machine_a = agent_multi_mutate()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, 3)
    result_b = run_to_completion(machine_b, 999)
    
    # Unwrap
    if hasattr(result_a, 'unwrap'):
        result_a = result_a.unwrap()
    if hasattr(result_b, 'unwrap'):
        result_b = result_b.unwrap()
    
    assert result_a == [1, 2, 3, 10, 20]
    assert result_b == [1, 2, 999, 10, 20]


def test_cow_nested_list_mutation():
    """Test that nested mutations (x[0].append(1)) trigger CoW on parent."""
    
    @compile_agent
    def agent_nested_mutate():
        # List of lists
        data = [[1, 2], [3, 4]]
        choice = branchpoint("choice")
        # Nested mutation: modify the inner list
        data[0].append(choice)
        return data
    
    machine_a = agent_nested_mutate()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, 100)
    result_b = run_to_completion(machine_b, 200)
    
    # Unwrap
    if hasattr(result_a, 'unwrap'):
        result_a = result_a.unwrap()
    if hasattr(result_b, 'unwrap'):
        result_b = result_b.unwrap()
    
    # Each branch should have isolated nested modifications
    assert result_a == [[1, 2, 100], [3, 4]], f"Branch A got {result_a}"
    assert result_b == [[1, 2, 200], [3, 4]], f"Branch B got {result_b}"


def test_cow_deeply_nested_mutation():
    """Test deeply nested mutations: x['users'][0]['items'].append(...)"""
    
    @compile_agent
    def agent_deeply_nested():
        state = {
            "users": [
                {"name": "Alice", "items": [1, 2]}
            ]
        }
        choice = branchpoint("choice")
        # Deep nested mutation
        state["users"][0]["items"].append(choice)
        return state
    
    machine_a = agent_deeply_nested()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, 42)
    result_b = run_to_completion(machine_b, 99)
    
    # Unwrap recursively
    def unwrap_deep(obj):
        if hasattr(obj, 'unwrap'):
            obj = obj.unwrap()
        if isinstance(obj, list):
            return [unwrap_deep(x) for x in obj]
        if isinstance(obj, dict):
            return {k: unwrap_deep(v) for k, v in obj.items()}
        return obj
    
    result_a = unwrap_deep(result_a)
    result_b = unwrap_deep(result_b)
    
    expected_a = {"users": [{"name": "Alice", "items": [1, 2, 42]}]}
    expected_b = {"users": [{"name": "Alice", "items": [1, 2, 99]}]}
    
    assert result_a == expected_a, f"Branch A got {result_a}"
    assert result_b == expected_b, f"Branch B got {result_b}"


def test_cow_augmented_assign_list():
    """Test that x += [items] correctly uses CoW proxy."""
    
    @compile_agent
    def agent_aug_assign():
        x = [1]
        choice = branchpoint("choice")
        x += [choice]  # This should trigger CoW via __iadd__
        return x
    
    machine_a = agent_aug_assign()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, 2)
    result_b = run_to_completion(machine_b, 999)
    
    # Unwrap
    if hasattr(result_a, 'unwrap'):
        result_a = result_a.unwrap()
    if hasattr(result_b, 'unwrap'):
        result_b = result_b.unwrap()
    
    # Each branch should have isolated results
    assert result_a == [1, 2], f"Branch A got {result_a}"
    assert result_b == [1, 999], f"Branch B got {result_b}"


def test_cow_aliasing():
    """Test that y = x doesn't create shared proxies (auto-unwrap on set)."""
    
    @compile_agent
    def agent_alias():
        x = [1, 2]
        y = x  # y should get a copy, not the same proxy
        choice = branchpoint("choice")
        if choice == "A":
            y.append(100)  # Modify y in branch A
        else:
            y.append(200)  # Modify y in branch B
        return (x, y)
    
    machine_a = agent_alias()
    sig = machine_a.run(None)
    machine_b = machine_a.snapshot()
    
    result_a = run_to_completion(machine_a, "A")
    result_b = run_to_completion(machine_b, "B")
    
    # Unwrap
    def unwrap(val):
        if hasattr(val, 'unwrap'):
            return val.unwrap()
        if isinstance(val, tuple):
            return tuple(unwrap(v) for v in val)
        return val
    
    result_a = unwrap(result_a)
    result_b = unwrap(result_b)
    
    # x should be unchanged, y should have the branch-specific modification
    # Both branches should be isolated
    assert result_a[1] == [1, 2, 100], f"Branch A y should be [1,2,100], got {result_a[1]}"
    assert result_b[1] == [1, 2, 200], f"Branch B y should be [1,2,200], got {result_b[1]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
