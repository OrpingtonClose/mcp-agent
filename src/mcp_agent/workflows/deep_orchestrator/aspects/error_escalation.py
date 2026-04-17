"""ErrorEscalationAspect -- graduated error handling by block criticality.

Ported from MiroThinker's ``aspects/error_escalation.py``.  Decides
whether to absorb or propagate errors based on each block's declared
``criticality`` (CRITICAL vs BEST_EFFORT).

- CRITICAL blocks: errors propagate (abort the pipeline).
- BEST_EFFORT blocks: errors are absorbed (logged, pipeline continues).
"""

from __future__ import annotations

import logging
from typing import Optional

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockCriticality,
    BlockResult,
    PipelineBlock,
    RoutingHint,
)

logger = logging.getLogger(__name__)


class ErrorEscalationAspect(Aspect):
    """Decide absorb vs propagate based on block criticality.

    Also tracks consecutive failures across all blocks and triggers
    EMERGENCY_STOP when the threshold is exceeded.
    """

    name = "error_escalation"

    def __init__(self, max_consecutive_failures: int = 3) -> None:
        self._max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0

    async def after(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        if result.metrics.get("block_failed") or result.metrics.get("error"):
            self._consecutive_failures += 1
            logger.warning(
                "Block '%s' failed (consecutive: %d/%d)",
                block.name,
                self._consecutive_failures,
                self._max_consecutive_failures,
            )

            if self._consecutive_failures >= self._max_consecutive_failures:
                logger.error(
                    "consecutive_failures=<%d> | escalating to "
                    "EMERGENCY_STOP",
                    self._consecutive_failures,
                )
                result.routing = RoutingHint.EMERGENCY_STOP
        else:
            self._consecutive_failures = 0

    async def on_error(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        error: Exception,
    ) -> Optional[BlockResult]:
        self._consecutive_failures += 1

        if block.criticality == BlockCriticality.CRITICAL:
            logger.error(
                "CRITICAL block '%s' failed, propagating error: %s",
                block.name,
                error,
            )
            # Return None to let the exception propagate
            return None

        # BEST_EFFORT: absorb the error, log, continue
        logger.warning(
            "BEST_EFFORT block '%s' failed (absorbed): %s",
            block.name,
            error,
        )
        routing = RoutingHint.CONTINUE
        if self._consecutive_failures >= self._max_consecutive_failures:
            routing = RoutingHint.EMERGENCY_STOP

        return BlockResult(
            metrics={"error": str(error), "block_failed": True, "absorbed": True},
            routing=routing,
            diagnosis=f"Block '{block.name}' error absorbed: {error}",
        )
