"""EnCompass: Continuous Program Search framework for LLM agents.

This module provides the public API for building search-based agents.
"""

from core.decorators import encompass_agent
from core.signals import BranchPoint, Effect, ScoreSignal, choose, effect
from encompass.std import action


# Helper functions
def branchpoint(name: str, **kwargs):
    """Create a branch point in the search space.

    Args:
        name: Identifier for this branch point
        **kwargs: Additional metadata for the branch point

    Returns:
        BranchPoint signal
    """
    return BranchPoint(name, kwargs)


def record_score(value: float, context: str = ""):
    """Record a score for the current execution path.

    Args:
        value: Score value (typically 0.0 to 1.0)
        context: Optional context describing what is being scored

    Returns:
        ScoreSignal containing the score
    """
    return ScoreSignal(value, context)


# Alias for compatibility with paper/docs
compile = encompass_agent  # noqa: A001
