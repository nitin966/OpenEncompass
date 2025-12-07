import unittest
import shutil
import os
from core.decorators import encompass_agent
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.strategies import MCTS, BeamSearch
from storage.filesystem import FileSystemStore

@encompass_agent
def simple_agent():
    # A simple agent for testing
    # Depth 0: 0 or 1
    choice = yield branchpoint("choice")
    if choice == 1:
        yield record_score(10.0)
        return "Win"
    else:
        yield record_score(0.0)
        return "Lose"

def simple_sampler(node):
    return [0, 1]

class TestEncompassCore(unittest.TestCase):
    def setUp(self):
        if os.path.exists("encompass_trace_test"):
            shutil.rmtree("encompass_trace_test")
        self.engine = ExecutionEngine()
        self.store = FileSystemStore(base_path="encompass_trace_test")

    def tearDown(self):
        if os.path.exists("encompass_trace_test"):
            shutil.rmtree("encompass_trace_test")

    def test_engine_step(self):
        root = self.engine.create_root()
        # Step with input 1 (Win)
        child, signal = self.engine.step(simple_agent, root, 1)
        self.assertTrue(child.is_terminal)
        self.assertEqual(child.score, 10.0)
        self.assertEqual(child.metadata['result'], "Win")
        
        # Step with input 0 (Lose)
        child_lose, _ = self.engine.step(simple_agent, root, 0)
        self.assertTrue(child_lose.is_terminal)
        self.assertEqual(child_lose.score, 0.0)

    def test_beam_search(self):
        beam = BeamSearch(self.store, self.engine, simple_sampler, width=2)
        results = beam.search(simple_agent)
        
        # Should find both paths
        self.assertTrue(len(results) >= 3)
        winning = [n for n in results if n.is_terminal and n.score == 10.0]
        self.assertEqual(len(winning), 1)

    def test_mcts_search(self):
        mcts = MCTS(self.store, self.engine, simple_sampler, iterations=50)
        results = mcts.search(simple_agent)
        
        # Should find the winning node
        winning_nodes = [n for n in results if n.is_terminal and n.score == 10.0]
        self.assertTrue(len(winning_nodes) > 0)

if __name__ == '__main__':
    unittest.main()
