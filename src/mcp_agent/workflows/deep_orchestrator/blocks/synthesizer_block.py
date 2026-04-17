"""SynthesizerBlock -- final synthesis phase with tool access.

Creates the final deliverable by aggregating all completed work,
accumulated knowledge, and artifacts into a comprehensive response.
"""

from __future__ import annotations

import logging
from collections import defaultdict
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
from mcp_agent.workflows.deep_orchestrator.prompts import (
    SYNTHESIZER_INSTRUCTION,
    get_synthesis_context,
    get_synthesis_prompt,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM, RequestParams

if TYPE_CHECKING:
    from mcp_agent.core.context import Context

logger = logging.getLogger(__name__)


class SynthesizerBlock(PipelineBlock):
    """Final synthesis of all work into a deliverable.

    Has tool access so it can read artifacts directly rather than
    relying on truncated text summaries.
    """

    name = "synthesizer"
    criticality = BlockCriticality.CRITICAL

    input_specs = [
        ParamSpec(
            key="objective",
            expected_type=str,
            description="the original objective",
        ),
        ParamSpec(
            key="completed_steps",
            expected_type=list,
            description="list of completed step data",
            required=False,
            default=[],
        ),
    ]

    output_specs = [
        ParamSpec(
            key="synthesis_output",
            expected_type=str,
            description="the final synthesized deliverable",
        ),
    ]

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM],
        available_servers: List[str],
        app_context: Optional["Context"] = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._available_servers = available_servers
        self._app_context = app_context

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Create the final deliverable from all work."""
        objective = ctx.state.get("objective", ctx.objective)
        completed_steps = ctx.state.get("completed_steps_data", [])
        knowledge_items = ctx.state.get("all_knowledge", [])
        artifacts = ctx.state.get("artifacts", {})
        budget_info = ctx.state.get("budget_info", {})

        # Give synthesizer tool access for reading artifacts directly
        synthesizer = Agent(
            name="FinalSynthesizer",
            instruction=SYNTHESIZER_INSTRUCTION,
            server_names=self._available_servers,
            context=self._app_context,
        )

        # Build execution summary
        execution_summary = {
            "iterations": ctx.state.get("iteration_count", 0),
            "steps_completed": len(completed_steps),
            "tasks_completed": ctx.state.get("tasks_completed_count", 0),
            "tokens_used": budget_info.get("tokens_used", 0),
            "cost": budget_info.get("cost_incurred", 0.0),
        }

        # Group knowledge by category
        knowledge_by_category: Dict[str, List[Any]] = defaultdict(list)
        for item in knowledge_items:
            category = getattr(item, "category", "general")
            knowledge_by_category[category].append(item)

        context = get_synthesis_context(
            objective=objective,
            execution_summary=execution_summary,
            completed_steps=completed_steps,
            knowledge_by_category=dict(knowledge_by_category),
            artifacts=artifacts,
        )

        prompt = get_synthesis_prompt(context)

        try:
            async with synthesizer:
                llm = await synthesizer.attach_llm(self._llm_factory)
                messages = await llm.generate(
                    message=prompt,
                    request_params=RequestParams(max_iterations=5),
                )

            # Extract text content from messages
            output = _extract_text_from_messages(messages)

            logger.info("Final synthesis completed (%d chars)", len(output))

            return BlockResult(
                metrics={
                    "synthesis_length": len(output),
                    "steps_synthesized": len(completed_steps),
                },
                state_updates={
                    "synthesis_output": output,
                    "synthesis_messages": messages,
                },
                routing=RoutingHint.FORCE_COMPLETE,
                output=output,
            )

        except Exception as e:
            logger.error("Synthesis failed: %s", e)
            return BlockResult(
                metrics={"error": str(e), "block_failed": True},
                routing=RoutingHint.FORCE_COMPLETE,
                diagnosis=f"Synthesis error: {e}",
            )


def _extract_text_from_messages(messages: list) -> str:
    """Extract text content from message objects.

    Fixes the original broken ``generate_str`` which did
    ``str(messages[0])`` and got a repr string instead of content.
    """
    if not messages:
        return ""

    parts = []
    for msg in messages:
        # Try common message content extraction patterns
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Content blocks (e.g. Anthropic format)
                for block in content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
            elif content is not None:
                parts.append(str(content))
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        parts.append(block["text"])
                    elif isinstance(block, str):
                        parts.append(block)
        elif isinstance(msg, str):
            parts.append(msg)
        else:
            # Last resort -- but at least try to get something useful
            msg_str = str(msg)
            if len(msg_str) < 10000:  # sanity check
                parts.append(msg_str)

    return "\n".join(parts)
