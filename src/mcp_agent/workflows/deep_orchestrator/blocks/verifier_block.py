"""VerifierBlock -- tool-equipped objective verification phase.

Unlike the original DeepOrchestrator's verifier (which had NO tools and
could only reason about text summaries), this block gives the verifier
access to MCP servers so it can actually inspect artifacts and validate
that work was completed correctly.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    BlockContext,
    BlockCriticality,
    BlockResult,
    ParamSpec,
    PipelineBlock,
    RoutingHint,
)
from mcp_agent.workflows.deep_orchestrator.models import VerificationResult
from mcp_agent.workflows.deep_orchestrator.prompts import (
    get_verification_context,
    get_verification_prompt,
)
from mcp_agent.workflows.llm.augmented_llm import AugmentedLLM

if TYPE_CHECKING:
    from mcp_agent.core.context import Context

logger = logging.getLogger(__name__)

TOOL_AWARE_VERIFIER_INSTRUCTION = """<verifier_instruction>
You are a thorough verifier who checks if objectives have been completed successfully.

<verification_process>
  <check>Has the core objective been achieved?</check>
  <check>Are all requested deliverables present?</check>
  <check>Is the quality sufficient for the intended purpose?</check>
  <check>Are there any critical gaps or missing elements?</check>
</verification_process>

<tool_awareness>
You have access to tools that let you inspect the actual workspace.
Use them to verify that artifacts were really created, that code compiles,
that files exist, etc.  Do NOT just reason about text summaries -- check
the actual state of the workspace.

<guidelines>
  <guideline>Verify files exist by listing/reading them</guideline>
  <guideline>Check that code changes are syntactically valid</guideline>
  <guideline>Validate that outputs match what was requested</guideline>
  <guideline>Report specific evidence for your assessment</guideline>
</guidelines>
</tool_awareness>

<assessment_criteria>
  <criterion>Completeness - all aspects addressed</criterion>
  <criterion>Correctness - accurate and valid results</criterion>
  <criterion>Quality - meets expected standards</criterion>
  <criterion>Usability - ready for intended use</criterion>
</assessment_criteria>

Be rigorous but fair. Consider partial success and acknowledge what has been achieved.
</verifier_instruction>"""


class VerifierBlock(PipelineBlock):
    """Tool-equipped verification with typed I/O contracts.

    Key improvement over original: the verifier agent has ``server_names``
    so it can actually inspect artifacts and validate correctness, rather
    than just reasoning about text summaries.
    """

    name = "verifier"
    criticality = BlockCriticality.BEST_EFFORT

    input_specs = [
        ParamSpec(
            key="objective",
            expected_type=str,
            description="the original objective to verify against",
        ),
        ParamSpec(
            key="progress_summary",
            expected_type=str,
            description="summary of work completed so far",
            required=False,
            default="",
        ),
    ]

    output_specs = [
        ParamSpec(
            key="verification_complete",
            expected_type=bool,
            description="whether the objective is verified complete",
        ),
        ParamSpec(
            key="verification_confidence",
            expected_type=float,
            description="confidence level of the verification (0-1)",
        ),
    ]

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM],
        available_servers: List[str],
        app_context: Optional["Context"] = None,
        min_confidence: float = 0.8,
    ) -> None:
        self._llm_factory = llm_factory
        self._available_servers = available_servers
        self._app_context = app_context
        self._min_confidence = min_confidence

    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Verify objective completion using tools to inspect actual state."""
        objective = ctx.state.get("objective", ctx.objective)
        progress_summary = ctx.state.get("progress_summary", "")
        knowledge_summary = ctx.state.get("knowledge_summary", "")
        artifacts = ctx.state.get("artifacts", {})

        # Give verifier actual tools to inspect the workspace.
        # Use async-with to ensure MCP connections are shut down.
        verifier = Agent(
            name="ObjectiveVerifier",
            instruction=TOOL_AWARE_VERIFIER_INSTRUCTION,
            server_names=self._available_servers[:3],
            context=self._app_context,
        )

        async with verifier:
            llm = await verifier.attach_llm(self._llm_factory)

            context = get_verification_context(
                objective=objective,
                progress_summary=progress_summary,
                knowledge_summary=knowledge_summary,
                artifacts=artifacts,
            )

            prompt = get_verification_prompt(context)

            try:
                result: VerificationResult = await llm.generate_structured(
                    message=prompt,
                    response_model=VerificationResult,
                )

                is_complete = (
                    result.is_complete and result.confidence >= self._min_confidence
                )

                logger.info(
                    "Verification: complete=%s, confidence=%.2f, missing=%d",
                    result.is_complete,
                    result.confidence,
                    len(result.missing_elements),
                )

                routing = RoutingHint.CONTINUE
                if is_complete:
                    routing = RoutingHint.FORCE_COMPLETE
                elif not result.is_complete:
                    routing = RoutingHint.REPLAN

                return BlockResult(
                    metrics={
                        "is_complete": result.is_complete,
                        "confidence": result.confidence,
                        "missing_elements": len(result.missing_elements),
                    },
                    state_updates={
                        "verification_complete": is_complete,
                        "verification_confidence": result.confidence,
                        "verification_reasoning": result.reasoning,
                        "missing_elements": result.missing_elements,
                    },
                    routing=routing,
                )

            except Exception as e:
                logger.error("Verification failed: %s", e)
                return BlockResult(
                    metrics={"error": str(e), "block_failed": True},
                    state_updates={
                        "verification_complete": False,
                        "verification_confidence": 0.0,
                    },
                    routing=RoutingHint.CONTINUE,
                    diagnosis=f"Verification error: {e}",
                )
