"""ExecutorBlock -- task execution phase without meta-prompting.

Replaces the original AgentDesigner meta-prompting pattern (where an LLM
designs a prompt for another LLM) with direct agent creation.  Each task
gets an Agent with the appropriate server_names and a focused instruction
derived from the task description -- no token-burning agent designer.
"""

from __future__ import annotations

import asyncio
import logging
import time
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
from mcp_agent.workflows.deep_orchestrator.models import (
    Step,
    TaskResult,
    TaskStatus,
)
from mcp_agent.workflows.deep_orchestrator.prompts import get_task_context
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams

if TYPE_CHECKING:
    from mcp_agent.core.context import Context

logger = logging.getLogger(__name__)


def _build_direct_instruction(task_description: str, servers: List[str]) -> str:
    """Build a focused instruction for a task agent without meta-prompting.

    Replaces the AgentDesigner pattern.  Instead of burning tokens to have
    an LLM design another LLM's prompt, we construct a clear instruction
    directly from the task description and available tools.
    """
    parts = [
        "<agent_instruction>",
        f"You are a task executor specialising in: {task_description}",
        "",
        "<execution_guidelines>",
        "  <guideline>Complete the task thoroughly using available tools</guideline>",
        "  <guideline>Be precise and detailed in your work</guideline>",
        "  <guideline>Report findings clearly with evidence</guideline>",
        "  <guideline>If you encounter errors, describe them "
        "and try alternatives</guideline>",
        "</execution_guidelines>",
    ]

    if servers:
        parts.append("")
        parts.append("<available_tools>")
        for server in servers:
            parts.append(f"  <tool>{server}</tool>")
        parts.append("</available_tools>")
        parts.append(
            "<important>Use these tools actively to complete your task</important>"
        )

    parts.append("</agent_instruction>")
    return "\n".join(parts)


class ExecutorBlock(PipelineBlock):
    """Execute a step's tasks with direct agent creation (no meta-prompting).

    Key improvements over original:
    1. No AgentDesigner -- agents are created directly with focused instructions
    2. Tasks get proper server_names so they have real tools
    3. Per-task timeout prevents infinite hangs (stall detection)
    """

    name = "executor"
    criticality = BlockCriticality.BEST_EFFORT

    input_specs = [
        ParamSpec(
            key="current_step",
            description="the Step to execute",
            required=True,
        ),
        ParamSpec(
            key="objective",
            expected_type=str,
            description="the user's objective for context",
        ),
    ]

    output_specs = [
        ParamSpec(
            key="step_results",
            expected_type=list,
            description="list of TaskResult objects",
        ),
    ]

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM],
        available_agents: Dict[str, Any],
        app_context: Optional["Context"] = None,
        max_task_retries: int = 3,
        task_timeout_seconds: float = 180.0,
        enable_parallel: bool = True,
    ) -> None:
        self._llm_factory = llm_factory
        self._available_agents = available_agents
        self._app_context = app_context
        self._max_task_retries = max_task_retries
        self._task_timeout = task_timeout_seconds
        self._enable_parallel = enable_parallel

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Execute all tasks in the current step."""
        step: Step = ctx.state.get("current_step")
        if step is None:
            return BlockResult(
                metrics={"error": "No step to execute"},
                routing=RoutingHint.CONTINUE,
            )

        objective = ctx.state.get("objective", ctx.objective)
        logger.info(
            "Executing step with %d tasks: %s",
            len(step.tasks),
            step.description[:80],
        )

        # Execute tasks (parallel or sequential)
        if self._enable_parallel and len(step.tasks) > 1:
            results = await asyncio.gather(
                *[
                    self._execute_task_with_timeout(task, objective)
                    for task in step.tasks
                ],
                return_exceptions=True,
            )
            # Convert exceptions to failed TaskResults
            task_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    task_results.append(
                        TaskResult(
                            task_name=step.tasks[i].name,
                            status=TaskStatus.FAILED,
                            error=str(result),
                        )
                    )
                else:
                    task_results.append(result)
        else:
            task_results = []
            for task in step.tasks:
                result = await self._execute_task_with_timeout(task, objective)
                task_results.append(result)

        successful = sum(1 for r in task_results if r.success)
        failed = len(task_results) - successful

        logger.info(
            "Step complete: %d successful, %d failed",
            successful,
            failed,
        )

        return BlockResult(
            metrics={
                "tasks_total": len(step.tasks),
                "tasks_successful": successful,
                "tasks_failed": failed,
                "step_description": step.description[:100],
            },
            state_updates={
                "step_results": task_results,
                "last_step_success": failed == 0,
            },
            routing=RoutingHint.CONTINUE,
        )

    async def _execute_task_with_timeout(
        self,
        task: Any,
        objective: str,
    ) -> TaskResult:
        """Execute a single task with timeout and retry logic."""
        if self._max_task_retries <= 0:
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.FAILED,
                error="max_task_retries is 0, no attempts made",
            )

        result: TaskResult | None = None
        for attempt in range(self._max_task_retries):
            try:
                result = await asyncio.wait_for(
                    self._execute_task_once(task, objective, attempt),
                    timeout=self._task_timeout,
                )
                if result.success:
                    return result
                if attempt < self._max_task_retries - 1:
                    logger.warning(
                        "Task '%s' failed (attempt %d/%d), retrying",
                        task.name,
                        attempt + 1,
                        self._max_task_retries,
                    )
                    await asyncio.sleep(2**attempt)
            except asyncio.TimeoutError:
                logger.error(
                    "Task '%s' timed out after %.0fs (attempt %d/%d)",
                    task.name,
                    self._task_timeout,
                    attempt + 1,
                    self._max_task_retries,
                )
                result = TaskResult(
                    task_name=task.name,
                    status=TaskStatus.FAILED,
                    error=f"Timed out after {self._task_timeout}s",
                    retry_count=attempt + 1,
                )
                if attempt < self._max_task_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                return result
            except Exception as e:
                logger.error("Task '%s' error: %s", task.name, e)
                if attempt == self._max_task_retries - 1:
                    return TaskResult(
                        task_name=task.name,
                        status=TaskStatus.FAILED,
                        error=str(e),
                        retry_count=attempt + 1,
                    )

        assert result is not None  # loop body always assigns result
        return result

    async def _execute_task_once(
        self,
        task: Any,
        objective: str,
        attempt: int,
    ) -> TaskResult:
        """Execute a single task attempt with direct agent creation."""
        start_time = time.time()

        try:
            agent = self._create_agent_directly(task)

            # Build task context
            task_context = get_task_context(
                objective=objective,
                task_description=task.description,
                required_servers=task.servers,
            )

            # Execute
            if isinstance(agent, AugmentedLLM):
                output = await agent.generate_str(
                    message=task_context,
                    request_params=RequestParams(max_iterations=10),
                )
            else:
                async with agent:
                    llm = await agent.attach_llm(self._llm_factory)
                    output = await llm.generate_str(
                        message=task_context,
                        request_params=RequestParams(max_iterations=10),
                    )

            duration = time.time() - start_time
            task.status = TaskStatus.COMPLETED

            logger.info(
                "Task '%s' completed in %.1fs",
                task.name,
                duration,
            )

            return TaskResult(
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                output=output,
                duration_seconds=duration,
                retry_count=attempt,
            )

        except Exception as e:
            duration = time.time() - start_time
            task.status = TaskStatus.FAILED
            logger.error("Task '%s' failed: %s", task.name, e)
            return TaskResult(
                task_name=task.name,
                status=TaskStatus.FAILED,
                error=str(e),
                duration_seconds=duration,
                retry_count=attempt,
            )

    def _create_agent_directly(self, task: Any) -> Agent:
        """Create an agent directly without meta-prompting.

        Replaces the AgentDesigner pattern entirely.  Instead of spending
        tokens asking an LLM to design another LLM's prompt, we construct
        a focused instruction from the task description.
        """
        # Check if a predefined agent was requested
        if task.agent and task.agent in self._available_agents:
            logger.debug("Using predefined agent: %s", task.agent)
            return self._available_agents[task.agent]

        # Create agent directly with focused instruction
        instruction = _build_direct_instruction(task.description, task.servers)

        return Agent(
            name=f"Executor_{task.name}",
            instruction=instruction,
            server_names=task.servers,
            context=self._app_context,
        )
