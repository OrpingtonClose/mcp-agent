"""HealthGateAspect -- evaluate block health and track cumulative state.

Ported from MiroThinker's ``aspects/health_gate.py``.  Tracks per-block
health metrics across iterations and flags degraded states without
imposing hard aborts (that's ErrorEscalationAspect's job).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockResult,
    PipelineBlock,
)

logger = logging.getLogger(__name__)


@dataclass
class _PhaseHealth:
    """Accumulated health for a single block across iterations."""

    total_runs: int = 0
    failures: int = 0
    last_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class HealthGateAspect(Aspect):
    """Track cumulative block health and evaluate simple gates.

    After each block execution, records success/failure and warns when
    the failure rate exceeds a threshold.  The health summary is stored
    in ``ctx.state["_block_health"]`` for downstream consumers (e.g.
    the PolicyEngine or synthesis prompts).
    """

    name = "health_gate"

    def __init__(self, failure_rate_threshold: float = 0.5) -> None:
        self._health: Dict[str, _PhaseHealth] = {}
        self._failure_rate_threshold = failure_rate_threshold

    def _get(self, name: str) -> _PhaseHealth:
        if name not in self._health:
            self._health[name] = _PhaseHealth()
        return self._health[name]

    async def after(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        ph = self._get(block.name)
        ph.total_runs += 1
        ph.last_metrics = dict(result.metrics)

        if result.metrics.get("error") or result.metrics.get("block_failed"):
            ph.failures += 1

        # Warn if failure rate exceeds threshold after 2+ runs
        if ph.total_runs >= 2:
            rate = ph.failures / ph.total_runs
            if rate > self._failure_rate_threshold:
                msg = (
                    f"Block '{block.name}' has >{self._failure_rate_threshold:.0%} "
                    f"failure rate ({ph.failures}/{ph.total_runs})"
                )
                if msg not in ph.warnings:
                    ph.warnings.append(msg)
                logger.warning(msg)

        # Persist health summary in state for downstream consumers
        ctx.state["_block_health"] = self.summary()

    def summary(self) -> Dict[str, Any]:
        """Return a serialisable health summary."""
        return {
            name: {
                "total_runs": ph.total_runs,
                "failures": ph.failures,
                "failure_rate": (
                    round(ph.failures / ph.total_runs, 2) if ph.total_runs > 0 else 0.0
                ),
                "warnings": ph.warnings[-5:],
            }
            for name, ph in self._health.items()
        }

    def get_block_failure_rate(self, block_name: str) -> float:
        """Get failure rate for a specific block."""
        ph = self._health.get(block_name)
        if ph is None or ph.total_runs == 0:
            return 0.0
        return ph.failures / ph.total_runs
