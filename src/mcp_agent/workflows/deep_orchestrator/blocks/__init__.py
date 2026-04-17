"""Pipeline blocks for the Deep Orchestrator.

Each block wraps a phase of the orchestrator's plan→execute→verify→synthesize
loop with typed I/O contracts (ParamSpec) and declared criticality levels.
"""

from mcp_agent.workflows.deep_orchestrator.blocks.planner_block import PlannerBlock
from mcp_agent.workflows.deep_orchestrator.blocks.executor_block import ExecutorBlock
from mcp_agent.workflows.deep_orchestrator.blocks.verifier_block import VerifierBlock
from mcp_agent.workflows.deep_orchestrator.blocks.synthesizer_block import (
    SynthesizerBlock,
)

__all__ = [
    "ExecutorBlock",
    "PlannerBlock",
    "SynthesizerBlock",
    "VerifierBlock",
]
