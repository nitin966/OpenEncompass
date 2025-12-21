"""
Adaptive Branching Monte Carlo Tree Search (AB-MCTS).

This module implements the AB-MCTS-A algorithm described in:
"Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search"
(Inoue et al., 2025).

The algorithm uses Thompson Sampling with Bayesian Conjugate Priors to adaptively balance:
- **Generation (Wider)**: Creating new child nodes from the current state.
- **Refinement (Deeper)**: Exploiting and exploring existing sub-trees.

This implementation emphasizes mathematical rigor and type safety.
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypeVar
from uuid import UUID

from core.signals import BranchPoint
from runtime.engine import ExecutionEngine
from runtime.node import SearchNode
from search.bayesian import (
    BetaPosterior,
    NormalInverseGammaPosterior,
    ProbabilisticPosterior,
)
from storage.base import StateStore

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ABMCTSConfig:
    r"""
    Configuration for AB-MCTS hyperparameters.

    Attributes:
        iterations: Total computational budget (number of expansions).
        score_type: "gaussian" (unbounded) or "beta" (bounded [0,1]).
        belief_sharing: "independent" (per-node) or "pooled" (shared Empirical Bayes).
        prior_mean: Prior mean ($\mu_0$) for Gaussian posteriors.
        prior_kappa: Prior confidence ($\kappa_0$) in mean.
        prior_alpha: Prior shape ($\alpha_0$) for variance.
        prior_beta: Prior scale ($\beta_0$) for variance.
        beta_prior_alpha: Prior alpha for Beta distribution.
        beta_prior_beta: Prior beta for Beta distribution.
    """

    iterations: int = 100
    score_type: str = "gaussian"
    
    # "independent" = AB-MCTS-A (Independent per-node posteriors).
    # "pooled"      = AB-MCTS-M (Shared knowledge via Empirical Bayes/Pooling).
    belief_sharing: str = "independent" 
    
    # Normal-Inverse-Gamma Priors
    prior_mean: float = 0.5
    prior_kappa: float = 1.0
    prior_alpha: float = 2.0
    prior_beta: float = 1.0

    # Beta Priors
    beta_prior_alpha: float = 1.0
    beta_prior_beta: float = 1.0


class SamplerProtocol(Protocol):
    """Protocol for the sampler callable."""
    async def __call__(self, node: SearchNode, metadata: dict[str, Any] | None) -> list[Any]: ...


@dataclass
class NodeState:
    """
    Probabilistic state attached to each search node.
    Maintains beliefs about the value of Generating vs Continuing.
    """
    gen_posterior: ProbabilisticPosterior
    cont_posterior: ProbabilisticPosterior
    child_ids: list[UUID] = field(default_factory=list)

    def add_child(self, child_id: UUID) -> None:
        self.child_ids.append(child_id)


class ABMCTS:
    """
    Adaptive Branching Monte Carlo Tree Search strategy.
    
    Executes a search using the AB-MCTS-A or AB-MCTS-M algorithm.
    """

    def __init__(
        self,
        store: StateStore | None,
        engine: ExecutionEngine,
        sampler: SamplerProtocol,
        config: ABMCTSConfig | None = None,
    ):
        self.store = store
        self.engine = engine
        self.sampler = sampler
        self.config = config or ABMCTSConfig()

        # Search Traversal State
        self.nodes: dict[UUID, SearchNode] = {}
        self.signals: dict[UUID, Any] = {}
        self.node_states: dict[UUID, NodeState] = {}
        self.parent_map: dict[UUID, UUID] = {}  # child_id -> parent_id

        # Global Posteriors (for AB-MCTS-M variant support)
        # These track the aggregate statistics of the entire tree.
        self.global_gen_posterior = self._create_base_posterior()
        self.global_cont_posterior = self._create_base_posterior()

    async def search(self, agent_factory: Callable) -> list[SearchNode]:
        """
        Run the search algorithm.

        Args:
            agent_factory: Factory function producing the root agent.

        Returns:
            List of all visited nodes.
        """
        logger.info(f"Starting AB-MCTS search (budget={self.config.iterations}, sharing={self.config.belief_sharing})")
        
        # 1. Initialize Root
        root = self.engine.create_root()
        root, signal = await self.engine.step(agent_factory, root)
        
        self._register_node(root, signal, parent_id=None)
        if self.store:
            self.store.save_node(root)

        # 2. Main Search Loop
        for i in range(self.config.iterations):
            # Selection Phase
            expand_node = self._select(root)
            
            # If selection hit a terminal node, backup its score
            if expand_node.is_terminal:
                self._backup(expand_node.node_id, expand_node.score)
                continue
            
            # Expansion Phase
            child = await self._expand(agent_factory, expand_node)
            
            # Backup Phase
            if child is not None:
                self._backup(child.node_id, child.score)
                
        logger.info(f"AB-MCTS finished. total_nodes={len(self.nodes)}")
        return list(self.nodes.values())

    def _create_base_posterior(self) -> ProbabilisticPosterior:
        """Create a posterior from static config (base priors)."""
        if self.config.score_type == "gaussian":
            return NormalInverseGammaPosterior(
                mu=self.config.prior_mean,
                kappa=self.config.prior_kappa,
                alpha=self.config.prior_alpha,
                beta=self.config.prior_beta,
            )
        elif self.config.score_type == "beta":
            return BetaPosterior(
                alpha=self.config.beta_prior_alpha,
                beta=self.config.beta_prior_beta,
            )
        else:
            raise ValueError(f"Unknown score_type: {self.config.score_type}")

    def _create_empirical_posterior(self, source: ProbabilisticPosterior) -> ProbabilisticPosterior:
        """
        Create a new posterior initialized with parameters from a source (Empirical Bayes).
        Used for AB-MCTS-M to inherit global knowledge.
        """
        if isinstance(source, NormalInverseGammaPosterior):
            return NormalInverseGammaPosterior(
                mu=source.mu,
                kappa=source.kappa,
                alpha=source.alpha,
                beta=source.beta,
            )
        elif isinstance(source, BetaPosterior):
            return BetaPosterior(
                alpha=source.alpha,
                beta=source.beta,
            )
        else:
            return self._create_base_posterior()

    def _register_node(self, node: SearchNode, signal: Any, parent_id: Optional[UUID]) -> None:
        """Register a new node in the internal state maps."""
        self.nodes[node.node_id] = node
        self.signals[node.node_id] = signal
        
        # Initialize beliefs for this node
        if self.config.belief_sharing == "pooled":
            # Mixed Model: Use Empirical Bayes (Global Prior)
            gen_post = self._create_empirical_posterior(self.global_gen_posterior)
            cont_post = self._create_empirical_posterior(self.global_cont_posterior)
        else:
            # Independent Model (A): Use Static Prior
            gen_post = self._create_base_posterior()
            cont_post = self._create_base_posterior()

        self.node_states[node.node_id] = NodeState(
            gen_posterior=gen_post,
            cont_posterior=cont_post,
        )

        if parent_id is not None:
            self.parent_map[node.node_id] = parent_id
            parent_state = self.node_states.get(parent_id)
            if parent_state:
                parent_state.add_child(node.node_id)
        
        if self.store:
            self.store.save_node(node)

    def _select(self, root: SearchNode) -> SearchNode:
        """
        Traverse the tree from the root to find a node to expand.
        """
        current = root
        
        while not current.is_terminal:
            state = self.node_states.get(current.node_id)
            if not state:
                logger.warning(f"Node {current.node_id} has no state. Stopping selection.")
                return current
            
            if not state.child_ids:
                return current
            
            # Thompson Sampling
            v_gen = state.gen_posterior.sample()
            v_cont = state.cont_posterior.sample()
            
            if v_gen > v_cont:
                return current
            else:
                best_child: Optional[SearchNode] = None
                best_val = float("-inf")
                
                candidates = []
                for cid in state.child_ids:
                    child = self.nodes.get(cid)
                    if not child: continue
                    
                    child_state = self.node_states.get(child.node_id)
                    
                    if child.is_terminal:
                        val = child.score
                    elif child_state:
                        val = child_state.cont_posterior.sample()
                    else:
                        val = 0.0
                        
                    candidates.append((val, child))
                
                if not candidates:
                    return current
                
                _, best_child = max(candidates, key=lambda x: x[0])
                if best_child:
                    current = best_child
                else:
                    return current
                    
        return current

    async def _expand(self, agent_factory: Callable, node: SearchNode) -> Optional[SearchNode]:
        """Generate a new child for the given node."""
        signal = self.signals.get(node.node_id)
        
        if not isinstance(signal, BranchPoint):
            return None
            
        meta = signal.metadata or {}
        
        # Sampler logic
        actions = await self.sampler(node, meta)
        
        if not actions:
            return None
            
        action = actions[0]
        
        child, new_signal = await self.engine.step(agent_factory, node, action)
        self._register_node(child, new_signal, parent_id=node.node_id)
        return child

    def _backup(self, node_id: UUID, score: float) -> None:
        """
        Backpropagate score to update Bayesian posteriors.
        Supports both Independent (A) and Mixed (M) variants.
        """
        val = score
        if self.config.score_type == "beta":
            val = max(0.0, min(1.0, score / 100.0))
            
        curr_id = node_id
        is_direct_child = True
        
        # Update Global Posteriors (if variant is M)
        # Note: We update both Gen and Cont global pool with every score?
        # Typically, a score reflects both the Gen decision (it was generated) 
        # and Cont potential (it's part of a path).
        # We'll simplify: update both to track "global average value".
        if self.config.belief_sharing == "pooled":
             self.global_gen_posterior.update(val)
             self.global_cont_posterior.update(val)
        
        while curr_id in self.parent_map:
            parent_id = self.parent_map[curr_id]
            parent_state = self.node_states.get(parent_id)
            
            if parent_state:
                if is_direct_child:
                    parent_state.gen_posterior.update(val)
                    is_direct_child = False
                else:
                    parent_state.cont_posterior.update(val)
            
            curr_id = parent_id
