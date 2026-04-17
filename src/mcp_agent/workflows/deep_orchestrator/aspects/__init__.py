"""Aspect-oriented cross-cutting concerns for the Deep Orchestrator.

Ported from MiroThinker's pipeline block framework. Aspects wrap every
phase of the orchestrator with uniform before/after/on_error hooks for
health tracking, budget management, error escalation, and timing.
"""

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    Aspect,
    BlockContext,
    BlockCriticality,
    BlockResult,
    ParamSpec,
    PipelineBlock,
    PipelineRunner,
    RoutingHint,
)
from mcp_agent.workflows.deep_orchestrator.aspects.budget_tracking import (
    BudgetTrackingAspect,
)
from mcp_agent.workflows.deep_orchestrator.aspects.error_escalation import (
    ErrorEscalationAspect,
)
from mcp_agent.workflows.deep_orchestrator.aspects.health_gate import (
    HealthGateAspect,
)
from mcp_agent.workflows.deep_orchestrator.aspects.timing import TimingAspect

__all__ = [
    "Aspect",
    "BlockContext",
    "BlockCriticality",
    "BlockResult",
    "BudgetTrackingAspect",
    "ErrorEscalationAspect",
    "HealthGateAspect",
    "ParamSpec",
    "PipelineBlock",
    "PipelineRunner",
    "RoutingHint",
    "TimingAspect",
]
