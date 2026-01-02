"""
Tests for tuple unpacking in CPS compiler.

These tests verify that tuple/list unpacking assignments 
correctly store values in _ctx.
"""

import pytest

from core.compiler import compile_agent
from core.signals import branchpoint


def test_simple_tuple_unpack():
    """Test basic tuple unpacking: x, y = (1, 2)"""
    
    @compile_agent
    def agent():
        x, y = (10, 20)
        z = branchpoint("check")
        return x + y + z
    
    machine = agent()
    sig = machine.run(None)
    assert sig.name == "check"
    
    # Check that x and y are in _ctx
    assert machine._ctx.get("x") == 10
    assert machine._ctx.get("y") == 20
    
    # Provide z and complete
    while not machine._done:
        machine.run(5)
    
    assert machine._result == 35  # 10 + 20 + 5


def test_tuple_unpack_across_branchpoint():
    """Test tuple unpacking where value comes from a branchpoint."""
    
    @compile_agent
    def agent():
        result = branchpoint("get_pair")
        x, y = result
        return x * y
    
    machine = agent()
    sig = machine.run(None)
    assert sig.name == "get_pair"
    
    # Provide a tuple
    while not machine._done:
        machine.run((3, 7))
    
    # Both x and y should be stored and used
    assert machine._result == 21  # 3 * 7


def test_list_unpack():
    """Test list unpacking: [a, b] = [1, 2]"""
    
    @compile_agent
    def agent():
        [a, b] = [100, 200]
        return a - b
    
    machine = agent()
    while not machine._done:
        machine.run(None)
    
    assert machine._result == -100


def test_nested_tuple_unpack():
    """Test nested tuple unpacking: (a, (b, c)) = (1, (2, 3))"""
    
    @compile_agent
    def agent():
        a, (b, c) = (1, (2, 3))
        return a + b + c
    
    machine = agent()
    while not machine._done:
        machine.run(None)
    
    assert machine._result == 6  # 1 + 2 + 3


def test_tuple_unpack_with_underscore():
    """Test tuple unpacking with ignored values: x, _ = (1, 2)"""
    
    @compile_agent
    def agent():
        x, _ = (42, 99)
        return x
    
    machine = agent()
    while not machine._done:
        machine.run(None)
    
    assert machine._result == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
