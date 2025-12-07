import pytest
from core.compiler import compile_agent as compile
from core.signals import branchpoint
from runtime.engine import ExecutionEngine

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@compile
def agent_with_try():
    try:
        x = yield branchpoint("choice")
        if x == 1:
            raise ValueError("Boom")
        return x
    except ValueError as e:
        return 999
    except Exception:
        return 0

@pytest.mark.anyio
async def test_try_except_handling():
    """
    Test basic try/except handling in CPS compiler.
    Verifies that exceptions raised in the agent are caught by the appropriate handler
    and that the agent can resume execution.
    """
    engine = ExecutionEngine()
    root = engine.create_root()
    
    # Path 1: No exception
    node1, sig1 = await engine.step(agent_with_try, root)
    assert sig1.name == "choice"
    
    # Resume with 2 (no exception)
    node2, sig2 = await engine.step(agent_with_try, node1, 2)
    assert node2.is_terminal
    assert node2.metadata['result'] == 2
    
    # Path 2: Exception
    # Resume with 1 (raises ValueError)
    node3, sig3 = await engine.step(agent_with_try, node1, 1)
    assert node3.is_terminal
    assert node3.metadata['result'] == 999  # Caught by except ValueError

@compile
def agent_nested_try():
    try:
        try:
            x = yield branchpoint("inner")
            if x == 1:
                raise ValueError("Inner Boom")
        except ValueError:
            # Catch inner, raise outer
            raise KeyError("Outer Boom")
    except KeyError:
        return 777
    return 0

@pytest.mark.anyio
async def test_nested_try():
    """
    Test nested try/except blocks.
    Verifies that exceptions propagate correctly through nested handlers.
    """
    engine = ExecutionEngine()
    root = engine.create_root()
    
    node1, sig1 = await engine.step(agent_nested_try, root)
    
    # Trigger inner exception -> caught -> raises outer -> caught
    node2, sig2 = await engine.step(agent_nested_try, node1, 1)
    assert node2.metadata['result'] == 777
