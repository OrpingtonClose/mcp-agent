"""BudgetTrackingAspect -- track token/cost/time budgets per block.

Wraps the existing ``SimpleBudget`` with aspect-level instrumentation
so that budget snapshots are taken before each block and deltas are
recorded after.  Triggers FORCE_COMPLETE when budget is critical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockResult,
    PipelineBlock,
    RoutingHint,
)
from mcp_agent.workflows.deep_orchestrator.budget import SimpleBudget

if TYPE_CHECKING:
    from mcp_agent.tracing.token_counter import TokenCounter

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
        token_counter: Optional[TokenCounter] = None,
        critical_threshold: float = 0.9,
    ) -> None:
        self._budget = budget
        self._token_counter = token_counter
        self._critical_threshold = critical_threshold

    async def _get_total_tokens(self) -> int:
        """Read total tokens from the app-level token counter."""
        if self._token_counter is None:
            return 0
        try:
            usage = await self._token_counter.get_app_usage()
            return usage.total_tokens if usage else 0
        except Exception:
            return 0

    async def before(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
    ) -> Optional[BlockResult]:
        # Snapshot current totals before execution
        ctx._cost_snapshot = self._budget.cost_incurred
        ctx._token_snapshot = await self._get_total_tokens()

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
        # Feed token delta from token_counter into SimpleBudget
        current_tokens = await self._get_total_tokens()
        token_delta = current_tokens - getattr(ctx, "_token_snapshot", 0)
        if token_delta > 0:
            self._budget.update_tokens(token_delta)
            logger.debug(
                "block=<%s>, token_delta=<%d> | budget updated",
                block.name,
                token_delta,
            )

        # Record budget delta
        cost_delta = self._budget.cost_incurred - getattr(ctx, "_cost_snapshot", 0.0)
        result.metrics["_budget_delta_tokens"] = token_delta
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
