"""
Cost tracking for LLM token usage and expenses.

This module provides centralized cost aggregation across search executions,
enabling production-grade cost monitoring and budgeting.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class CostData:
    """
    Cost data for a single node execution.

    Attributes:
        tokens_in: Input tokens consumed
        tokens_out: Output tokens generated
        cost_usd: Total cost in USD
        timestamp: When the cost was recorded
        model: Model used (e.g., "gpt-4")
    """

    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: datetime
    model: str | None = None


class CostAggregator:
    """
    Tracks token usage and costs across all nodes in a search.

    Usage:
        aggregator = CostAggregator()
        aggregator.record(node_id, tokens_in=100, tokens_out=50, cost_usd=0.01)
        total = aggregator.get_total_cost()
    """

    def __init__(self):
        self.node_costs: dict[str, CostData] = {}  # node_id -> CostData

    def record(
        self,
        node_id: str,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        model: str | None = None,
    ):
        """
        Record cost for a specific node.

        Args:
            node_id: UUID of the node
            tokens_in: Input tokens consumed
            tokens_out: Output tokens generated
            cost_usd: Total cost in USD
            model: Model name (optional)
        """
        self.node_costs[node_id] = CostData(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            timestamp=datetime.now(),
            model=model,
        )

    def get_node_cost(self, node_id: str) -> CostData | None:
        """Get cost data for a specific node."""
        return self.node_costs.get(node_id)

    def get_total_cost(self) -> float:
        """Get total cost across all nodes in USD."""
        return sum(data.cost_usd for data in self.node_costs.values())

    def get_total_tokens(self) -> dict[str, int]:
        """Get total token usage across all nodes."""
        total_in = sum(data.tokens_in for data in self.node_costs.values())
        total_out = sum(data.tokens_out for data in self.node_costs.values())
        return {
            "tokens_in": total_in,
            "tokens_out": total_out,
            "tokens_total": total_in + total_out,
        }

    def get_summary(self) -> dict:
        """
        Get comprehensive cost summary.

        Returns:
            Dict with total_cost, total_tokens, node_count, and per_model breakdown
        """
        tokens = self.get_total_tokens()

        # Per-model breakdown
        model_breakdown = {}
        for data in self.node_costs.values():
            model = data.model or "unknown"
            if model not in model_breakdown:
                model_breakdown[model] = {
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "cost_usd": 0.0,
                    "count": 0,
                }
            model_breakdown[model]["tokens_in"] += data.tokens_in
            model_breakdown[model]["tokens_out"] += data.tokens_out
            model_breakdown[model]["cost_usd"] += data.cost_usd
            model_breakdown[model]["count"] += 1

        return {
            "total_cost_usd": self.get_total_cost(),
            "total_tokens": tokens,
            "node_count": len(self.node_costs),
            "models": model_breakdown,
        }

    def reset(self):
        """Clear all recorded costs."""
        self.node_costs.clear()
