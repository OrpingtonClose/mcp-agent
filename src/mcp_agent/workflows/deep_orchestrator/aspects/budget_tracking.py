"""BudgetTrackingAspect -- track token/cost/time budgets per block.

Wraps the existing ``SimpleBudget`` with aspect-level instrumentation
so that budget snapshots are taken before each block and deltas are
recorded after.  Triggers FORCE_COMPLETE when budget is critical.
"""

from __future__ import annotations

import logging
from typing import Optional

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockResult,
    PipelineBlock,
    RoutingHint,
)
from mcp_agent.workflows.deep_orchestrator.budget import SimpleBudget

logger = logging.getLogger(__name__)


class BudgetTrackingAspect(Aspect):
    """Track budget consumption per block via before/after snapshots.

    Stores ``_budget_delta_tokens`` and ``_budget_delta_cost`` in the
    result metrics so that callers can see per-block resource usage.
    """

    name = "budget_tracking"

    def __init__(
        self,
        budget: SimpleBudget,
        critical_threshold: float = 0.9,
    ) -> None:
        self._budget = budget
        self._critical_threshold = critical_threshold

    async def before(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
    ) -> Optional[BlockResult]:
        # Snapshot current budget before execution
        ctx._cost_snapshot = self._budget.cost_incurred

        # Check if budget is already exceeded
        exceeded, reason = self._budget.is_exceeded()
        if exceeded:
            logger.warning(
                "Budget exceeded before block '%s': %s",
                block.name,
                reason,
            )
            return BlockResult(
                metrics={"budget_exceeded": True, "reason": reason},
                routing=RoutingHint.FORCE_COMPLETE,
                diagnosis=f"Budget exceeded: {reason}",
            )

        # Check if approaching critical
        if self._budget.is_critical(self._critical_threshold):
            usage = self._budget.get_usage_pct()
            logger.warning(
                "Budget approaching critical before block '%s': %s",
                block.name,
                usage,
            )

        return None

    async def after(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        # Record budget delta
        cost_delta = self._budget.cost_incurred - ctx._cost_snapshot
        result.metrics["_budget_delta_cost"] = round(cost_delta, 4)
        result.metrics["_budget_tokens_total"] = self._budget.tokens_used
        result.metrics["_budget_cost_total"] = round(self._budget.cost_incurred, 4)

        # Check if budget is now exceeded after execution
        exceeded, reason = self._budget.is_exceeded()
        if exceeded:
            logger.warning(
                "Budget exceeded after block '%s': %s",
                block.name,
                reason,
            )
            result.routing = RoutingHint.FORCE_COMPLETE
