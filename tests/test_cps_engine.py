import unittest
from runtime.engine import ExecutionEngine
from core.compiler import compile_agent
from encompass import branchpoint

class TestCPSEngine(unittest.IsolatedAsyncioTestCase):
    async def test_cps_execution(self):
        # Define agent
        def my_agent():
            x = yield branchpoint("step1")
            y = yield branchpoint("step2")
            return x + y
            
        # Compile it
        AgentClass = compile_agent(my_agent)
        print(f"DEBUG: AgentClass bases: {AgentClass.__bases__}")
        print(f"DEBUG: AgentClass dir: {dir(AgentClass)}")
        
        engine = ExecutionEngine()
        root = engine.create_root()
        
        # Step 1: Start (should return step1 signal)
        node1, sig1 = await engine.step(AgentClass, root)
        self.assertEqual(sig1.name, "step1")
        self.assertIsNotNone(node1.machine_state)
        self.assertEqual(node1.machine_state['_state'], 1)
        
        # Step 2: Resume with input 10
        # Should NOT replay from start, but load state 1 and continue
        node2, sig2 = await engine.step(AgentClass, node1, 10)
        self.assertEqual(sig2.name, "step2")
        self.assertEqual(node2.machine_state['_state'], 2)
        self.assertEqual(node2.machine_state['_ctx']['x'], 10)
        
        # Step 3: Resume with input 20
        node3, sig3 = await engine.step(AgentClass, node2, 20)
        self.assertTrue(node3.is_terminal)
        self.assertEqual(node3.metadata['result'], 30)

if __name__ == "__main__":
    unittest.main()
