"""
Unit tests for strict AB-MCTS implementation.

Tests:
1. NormalInverseGammaPosterior (updates, rigor)
2. BetaPosterior
3. ABMCTS Search Flow with ABMCTSConfig
"""

import asyncio
import math
import unittest
from uuid import uuid4

from core.decorators import encompass_agent
from core.signals import branchpoint, record_score
from runtime.engine import ExecutionEngine
from search.ab_mcts import ABMCTS, ABMCTSConfig, NodeState
from search.bayesian import NormalInverseGammaPosterior, BetaPosterior
from storage.filesystem import FileSystemStore


# Define agents
@encompass_agent
def simple_agent():
    choice = branchpoint("choice")
    if choice == "option_a":
        record_score(80)
    else:
        record_score(40)
    return choice

@encompass_agent
def scored_agent():
    c = branchpoint("c")
    if c == "option_a":
        record_score(100)
        return "A"
    else:
        record_score(10)
        return "B"

@encompass_agent
def test_agent():
    branchpoint("step")
    record_score(75)
    return "done"


class TestNormalInverseGammaPosterior(unittest.TestCase):
    """Tests for NIG Posterior implementation rigor."""

    def test_update_logic(self):
        """Verify update formulas follow Murphy (2012) for n=1."""
        # Prior
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 2.0, 1.0
        post = NormalInverseGammaPosterior(mu=mu0, kappa=kappa0, alpha=alpha0, beta=beta0)
        
        # Observation
        x = 2.0
        post.update(x)
        
        # Expected N=1 updates
        # kappa_n = 1 + 1 = 2
        self.assertEqual(post.kappa, 2.0)
        
        # mu_n = (1*0.0 + 2.0) / 2 = 1.0
        self.assertEqual(post.mu, 1.0)
        
        # alpha_n = 2.0 + 0.5 = 2.5
        self.assertEqual(post.alpha, 2.5)
        
        # beta_n = 1.0 + (1.0 * (2.0 - 0.0)^2) / (2 * 2.0)
        # beta_n = 1.0 + (4.0) / 4.0 = 2.0
        self.assertEqual(post.beta, 2.0)

    def test_sample_variance(self):
        """Test sampling produces reasonable values."""
        post = NormalInverseGammaPosterior()
        samples = [post.sample() for _ in range(100)]
        
        # Check simple bounds (probability of extremely large/small values is low)
        avg = sum(samples) / len(samples)
        self.assertTrue(-5.0 < avg < 5.0)


class TestBetaPosterior(unittest.TestCase):
    """Tests for Beta Posterior."""
    
    def test_update(self):
        post = BetaPosterior(alpha=1, beta=1)
        post.update(0.8) # success-ish
        self.assertEqual(post.alpha, 1.8)
        self.assertEqual(post.beta, 1.2)


class TestABMCTS(unittest.TestCase):
    """Tests for AB-MCTS Search Strategy."""
    
    def setUp(self):
        self.engine = ExecutionEngine()
        self.store = FileSystemStore("output/test_ab_mcts_rigor")
        self.config = ABMCTSConfig(
            iterations=5, 
            score_type="gaussian",
            prior_mean=0.0
        )
    
    async def _sampler(self, node, metadata=None):
        return ["option_a", "option_b"]
    
    def test_search_completes(self):
        async def run():
            strategy = ABMCTS(
                store=self.store,
                engine=self.engine,
                sampler=self._sampler,
                config=self.config
            )
            return await strategy.search(simple_agent)
        
        results = asyncio.run(run())
        self.assertGreater(len(results), 0)
    
    def test_finds_high_scoring_path(self):
        # Config with enough iterations
        config = ABMCTSConfig(iterations=15, score_type="gaussian")
        
        async def run():
            strategy = ABMCTS(
                store=self.store,
                engine=self.engine,
                sampler=self._sampler,
                config=config
            )
            return await strategy.search(scored_agent)
        
        results = asyncio.run(run())
        best = max(results, key=lambda n: n.score)
        self.assertGreaterEqual(best.score, 50)
    
    def test_posteriors_updated(self):
        async def run():
            strategy = ABMCTS(
                store=self.store,
                engine=self.engine,
                sampler=self._sampler,
                config=self.config
            )
            await strategy.search(test_agent)
            
            total_obs = 0
            for state in strategy.node_states.values():
                total_obs += state.gen_posterior.n_observations
                total_obs += state.cont_posterior.n_observations
            return total_obs
        
        obs = asyncio.run(run())
        self.assertGreater(obs, 0)
    
    def test_mixed_model_sharing(self):
        """Test that belief_sharing='pooled' updates global priors."""
        config = ABMCTSConfig(
            iterations=5, 
            score_type="gaussian",
            belief_sharing="pooled",
            prior_mean=0.0
        )
        
        async def run():
            strategy = ABMCTS(
                store=self.store,
                engine=self.engine,
                sampler=self._sampler,
                config=config
            )
            # Run search
            await strategy.search(simple_agent)
            
            # Check global posterior updated
            self.assertGreater(strategy.global_gen_posterior.n_observations, 0)
            
            # Check that a new node (e.g. root wasn't, but subsequent ones)
            # would inherit non-zero parameters. 
            # We can inspect the last created node's initial params.
            # But simpler checks: global posterior has changed.
            return strategy.global_gen_posterior.kappa
            
        kappa = asyncio.run(run())
        # kappa starts at 1.0, should increase with observations
        self.assertGreater(kappa, 1.0)



if __name__ == "__main__":
    unittest.main()
