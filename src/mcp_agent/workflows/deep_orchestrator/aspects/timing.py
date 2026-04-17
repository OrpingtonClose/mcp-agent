"""TimingAspect -- wall-clock measurement and stall detection per block.

Ported from MiroThinker's ``aspects/timing.py``.  Measures execution
time for every block and detects stalls when a block exceeds its
timeout threshold.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockResult,
    PipelineBlock,
    RoutingHint,
)

logger = logging.getLogger(__name__)


class TimingAspect(Aspect):
    """Measure wall-clock time per block and detect stalls.

    Records ``_duration_seconds`` in the result metrics.  If a block
    exceeds ``stall_timeout_seconds``, logs a warning and optionally
    sets the routing hint to FORCE_COMPLETE.
    """

    name = "timing"

    def __init__(
        self,
        stall_timeout_seconds: float = 300.0,
        force_complete_on_stall: bool = False,
    ) -> None:
        self._stall_timeout = stall_timeout_seconds
        self._force_complete_on_stall = force_complete_on_stall

    async def before(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
    ) -> Optional[BlockResult]:
        ctx._phase_start_time = time.monotonic()
        return None

    async def after(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        if ctx._phase_start_time > 0:
            duration = time.monotonic() - ctx._phase_start_time
            result.metrics["_duration_seconds"] = round(duration, 2)

            if duration > self._stall_timeout:
                logger.warning(
                    "Block '%s' exceeded stall timeout: %.1fs > %.1fs",
                    block.name,
                    duration,
                    self._stall_timeout,
                )
                result.metrics["_stall_detected"] = True
                if self._force_complete_on_stall:
                    result.routing = RoutingHint.FORCE_COMPLETE
            else:
                logger.debug(
                    "Block '%s' completed in %.2fs",
                    block.name,
                    duration,
                )
