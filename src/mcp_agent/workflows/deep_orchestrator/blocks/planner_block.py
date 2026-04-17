"""PlannerBlock -- tool-equipped strategic planning phase.

Unlike the original DeepOrchestrator's planner (which had NO tools and
could only hallucinate server names), this block gives the planner access
to the available MCP servers so it can actually inspect the workspace
before creating a plan.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    BlockContext,
    BlockCriticality,
    BlockResult,
    ParamSpec,
    PipelineBlock,
    RoutingHint,
)
from mcp_agent.workflows.deep_orchestrator.models import Plan
from mcp_agent.workflows.deep_orchestrator.plan_verifier import PlanVerifier
from mcp_agent.workflows.deep_orchestrator.prompts import (
    PLANNER_INSTRUCTION,
    get_full_plan_prompt,
    get_planning_context,
)
from mcp_agent.workflows.deep_orchestrator.utils import retry_with_backoff
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

if TYPE_CHECKING:
    from mcp_agent.core.context import Context

logger = logging.getLogger(__name__)

# Enhanced planner instruction that is tool-aware
TOOL_AWARE_PLANNER_INSTRUCTION = (
    PLANNER_INSTRUCTION
    + """

<tool_awareness>
You have access to tools that let you inspect the workspace before planning.
Use them to understand what files exist, what code looks like, and what
resources are available.  This ensures your plan is grounded in reality
rather than assumptions.

<guidelines>
  <guideline>Before planning, use tools to verify assumptions
  about the workspace</guideline>
  <guideline>Check that referenced files/resources actually
  exist</guideline>
  <guideline>Understand the structure of the codebase before
  decomposing tasks</guideline>
</guidelines>
</tool_awareness>"""
)


class PlannerBlock(PipelineBlock):
    """Tool-equipped planning phase with typed I/O contracts.

    Key improvement over original: the planner agent has ``server_names``
    so it can actually inspect the workspace before creating a plan.
    """

    name = "planner"
    criticality = BlockCriticality.CRITICAL

    input_specs = [
        ParamSpec(
            key="objective",
            expected_type=str,
            description="the user's objective to plan for",
            validator=lambda v: len(v) > 0,
        ),
        ParamSpec(
            key="available_servers",
            expected_type=list,
            description="list of MCP server names",
            required=False,
            default=[],
        ),
    ]

    output_specs = [
        ParamSpec(
            key="plan",
            description="the structured execution plan",
            required=True,
        ),
    ]

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM],
        available_servers: List[str],
        available_agents: Dict[str, Any],
        app_context: Optional["Context"] = None,
        max_verification_attempts: int = 5,
    ) -> None:
        self._llm_factory = llm_factory
        self._available_servers = available_servers
        self._available_agents = available_agents
        self._app_context = app_context
        self._max_verification_attempts = max_verification_attempts
        self._plan_verifier = PlanVerifier(
            available_servers=available_servers,
            available_agents=available_agents,
        )

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Create a comprehensive execution plan.

        The planner now has server_names so it can inspect the workspace
        before creating its plan, rather than guessing at what exists.
        """
        objective = ctx.state.get("objective", ctx.objective)
        completed_steps = ctx.state.get("completed_step_descriptions", [])[-5:]
        knowledge_items = ctx.state.get("knowledge_items_for_planning", [])[:10]

        # Give the planner actual tools so it can inspect before planning.
        # Use async-with to ensure MCP connections are shut down.
        planner = Agent(
            name="StrategicPlanner",
            instruction=TOOL_AWARE_PLANNER_INSTRUCTION,
            server_names=self._available_servers[:3],
            context=self._app_context,
        )

        async with planner:
            llm = await planner.attach_llm(self._llm_factory)

            # Iterative plan creation with verification
            previous_plan: Optional[Plan] = None
            previous_errors = None

            for attempt in range(self._max_verification_attempts):
                context = get_planning_context(
                    objective=objective,
                    progress_summary=ctx.state.get("progress_summary", ""),
                    completed_steps=completed_steps,
                    knowledge_items=knowledge_items,
                    available_servers=self._available_servers,
                    available_agents=self._available_agents,
                )

                if previous_plan and previous_errors:
                    context += "\n\n<previous_failed_plan>\n"
                    context += previous_plan.model_dump_json(indent=2)
                    context += "\n</previous_failed_plan>"
                    context += (
                        f"\n\n<plan_errors>\n"
                        f"{previous_errors.get_error_summary()}"
                        f"\n</plan_errors>"
                    )

                prompt = get_full_plan_prompt(context)
                plan: Plan = await retry_with_backoff(
                    lambda: llm.generate_structured(
                        message=prompt, response_model=Plan
                    ),
                    max_attempts=2,
                )

                verification_result = self._plan_verifier.verify_plan(plan)

                if verification_result.is_valid:
                    logger.info(
                        "Created valid plan: %d steps, reasoning: %s",
                        len(plan.steps),
                        plan.reasoning[:100],
                    )
                    return BlockResult(
                        metrics={
                            "plan_steps": len(plan.steps),
                            "plan_tasks": sum(len(s.tasks) for s in plan.steps),
                            "verification_attempts": attempt + 1,
                        },
                        state_updates={
                            "current_plan": plan,
                            "plan_is_complete": plan.is_complete,
                        },
                        routing=(
                            RoutingHint.FORCE_COMPLETE
                            if plan.is_complete
                            else RoutingHint.CONTINUE
                        ),
                    )

                logger.warning(
                    "Plan verification failed (attempt %d/%d): %d errors",
                    attempt + 1,
                    self._max_verification_attempts,
                    len(verification_result.errors),
                )
                previous_plan = plan
                previous_errors = verification_result

        # All attempts exhausted -- return last plan with warning
        logger.error(
            "Failed to create valid plan after %d attempts",
            self._max_verification_attempts,
        )
        return BlockResult(
            metrics={
                "plan_steps": len(plan.steps) if plan else 0,
                "verification_attempts": (self._max_verification_attempts),
                "plan_validation_failed": True,
            },
            state_updates={
                "current_plan": plan,
                "plan_is_complete": False,
            },
            routing=RoutingHint.CONTINUE,
            diagnosis="Plan verification failed after max attempts",
        )
