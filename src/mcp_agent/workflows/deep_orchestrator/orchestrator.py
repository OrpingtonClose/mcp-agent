"""Deep Orchestrator - Aspect-oriented adaptive workflow orchestration.

Rewritten with MiroThinker-inspired patterns:
- Aspect-oriented pipeline (health gate, error escalation, timing, budget)
- Tool-equipped planner/verifier (no more hallucinating server names)
- Direct agent creation (no meta-prompting AgentDesigner)
- Knowledge store with dedup, scoring, contradiction detection
- Typed I/O contracts (ParamSpec) on every phase
- Per-task timeout / stall detection
- Fixed generate_str (proper content extraction, not str(messages[0]))

Architecture::

    DeepOrchestrator (AugmentedLLM interface preserved)
    └── PipelineRunner
          ├── aspects: [TimingAspect, BudgetTrackingAspect,
          │              HealthGateAspect, ErrorEscalationAspect]
          └── blocks:  [PlannerBlock, ExecutorBlock,
                        VerifierBlock, SynthesizerBlock]

    The while-loop manages sequencing.
    Each iteration delegates to runner.run_block() for the active phase.
"""

import time
from typing import Callable, List, Optional, Type, TYPE_CHECKING

from mcp_agent.agents.agent import Agent
from mcp_agent.logging.logger import get_logger
from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.tracing.token_tracking_decorator import track_tokens
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageParamT,
    MessageT,
    ModelT,
    RequestParams,
)

from mcp_agent.workflows.deep_orchestrator.aspects.base import (
    BlockContext,
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
from mcp_agent.workflows.deep_orchestrator.blocks.executor_block import (
    ExecutorBlock,
)
from mcp_agent.workflows.deep_orchestrator.blocks.planner_block import (
    PlannerBlock,
)
from mcp_agent.workflows.deep_orchestrator.blocks.synthesizer_block import (
    SynthesizerBlock,
)
from mcp_agent.workflows.deep_orchestrator.blocks.verifier_block import (
    VerifierBlock,
)
from mcp_agent.workflows.deep_orchestrator.budget import SimpleBudget
from mcp_agent.workflows.deep_orchestrator.config import DeepOrchestratorConfig
from mcp_agent.workflows.deep_orchestrator.knowledge_store import KnowledgeStore
from mcp_agent.workflows.deep_orchestrator.memory import WorkspaceMemory
from mcp_agent.workflows.deep_orchestrator.models import (
    KnowledgeItem,
    Plan,
)
from mcp_agent.workflows.deep_orchestrator.prompts import (
    EMERGENCY_RESPONDER_INSTRUCTION,
    ORCHESTRATOR_SYSTEM_INSTRUCTION,
    get_emergency_context,
    get_emergency_prompt,
)
from mcp_agent.workflows.deep_orchestrator.queue import TodoQueue

if TYPE_CHECKING:
    from opentelemetry.trace.span import Span
    from mcp_agent.core.context import Context

logger = get_logger(__name__)


class DeepOrchestrator(AugmentedLLM[MessageParamT, MessageT]):
    """Aspect-oriented adaptive orchestrator for deep research-style tasks.

    Coordinates specialised agents and MCP servers through:
    plan → execute → verify → replan → synthesise

    All phases are wrapped by the PipelineRunner's aspect stack so that
    cross-cutting concerns (health tracking, budget enforcement, error
    escalation, timing) are applied uniformly without cluttering the
    business logic of each phase.

    Backward-compatible: the public AugmentedLLM interface (generate,
    generate_str, generate_structured) is preserved for AetherAgent.
    """

    def __init__(
        self,
        llm_factory: Callable[[Agent], AugmentedLLM[MessageParamT, MessageT]],
        config: Optional[DeepOrchestratorConfig] = None,
        context: Optional["Context"] = None,
        **kwargs,
    ):
        if config is None:
            config = DeepOrchestratorConfig()

        super().__init__(
            name=config.name,
            instruction=ORCHESTRATOR_SYSTEM_INSTRUCTION,
            context=context,
            **kwargs,
        )

        self.llm_factory = llm_factory
        self.config = config
        self.agents = {agent.name: agent for agent in config.available_agents}

        # Discover available MCP servers
        if config.available_servers:
            self.available_servers = config.available_servers
        elif context and hasattr(context, "server_registry"):
            self.available_servers = list(context.server_registry.registry.keys())
            logger.info(
                "server_count=<%d> | detected MCP servers from registry",
                len(self.available_servers),
            )
        else:
            self.available_servers = []
            logger.warning("server_count=<0> | no MCP servers available")

        # Core components
        self.memory = WorkspaceMemory(
            use_filesystem=self.config.execution.enable_filesystem,
        )
        self.knowledge = KnowledgeStore(max_items=200)
        self.queue = TodoQueue()
        self.budget = SimpleBudget(
            max_tokens=self.config.budget.max_tokens,
            max_cost=self.config.budget.max_cost,
            max_time_minutes=self.config.budget.max_time_minutes,
            cost_per_1k_tokens=self.config.budget.cost_per_1k_tokens,
        )

        # Tracking
        self.objective: str = ""
        self.iteration: int = 0
        self.replan_count: int = 0
        self.start_time: float = 0.0
        self.current_plan: Optional[Plan] = None

        # Pipeline components (initialised per-execution in _build_pipeline)
        self._runner: Optional[PipelineRunner] = None
        self._health_gate: Optional[HealthGateAspect] = None

        logger.info(
            "name=<%s>, agents=<%d>, servers=<%d>, max_iter=<%d> | "
            "initialised DeepOrchestrator",
            config.name,
            len(self.agents),
            len(self.available_servers),
            config.execution.max_iterations,
        )

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> PipelineRunner:
        """Construct the aspect-wrapped pipeline blocks.

        Called once per execution (after the objective is known).
        """
        # --- Blocks ---
        planner = PlannerBlock(
            llm_factory=self.llm_factory,
            available_servers=self.available_servers,
            available_agents=self.agents,
            app_context=self.context,
            max_verification_attempts=min(5, self.config.execution.max_iterations),
        )

        executor = ExecutorBlock(
            llm_factory=self.llm_factory,
            available_agents=self.agents,
            app_context=self.context,
            max_task_retries=self.config.execution.max_task_retries,
            task_timeout_seconds=180.0,
            enable_parallel=self.config.execution.enable_parallel,
        )

        verifier = VerifierBlock(
            llm_factory=self.llm_factory,
            available_servers=self.available_servers,
            app_context=self.context,
            min_confidence=self.config.policy.min_verification_confidence,
        )

        synthesizer = SynthesizerBlock(
            llm_factory=self.llm_factory,
            available_servers=self.available_servers,
            app_context=self.context,
        )

        blocks = [planner, executor, verifier, synthesizer]

        # --- Aspects (order matters: timing first, escalation last) ---
        timing = TimingAspect(
            stall_timeout_seconds=300.0,
            force_complete_on_stall=False,
        )
        token_counter = (
            self.context.token_counter
            if self.context and hasattr(self.context, "token_counter")
            else None
        )
        budget = BudgetTrackingAspect(
            budget=self.budget,
            token_counter=token_counter,
            critical_threshold=self.config.policy.budget_critical_threshold,
        )
        self._health_gate = HealthGateAspect(failure_rate_threshold=0.5)
        escalation = ErrorEscalationAspect(
            max_consecutive_failures=self.config.policy.max_consecutive_failures,
        )

        aspects = [timing, budget, self._health_gate, escalation]

        return PipelineRunner(blocks=blocks, aspects=aspects)

    # ------------------------------------------------------------------
    # Public AugmentedLLM interface
    # ------------------------------------------------------------------

    @track_tokens(node_type="workflow")
    async def generate(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> List[MessageT]:
        """Main execution entry point (AugmentedLLM interface)."""
        tracer = get_tracer(self.context)

        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.generate"
        ) as span:
            # Extract objective
            if isinstance(message, str):
                objective = message
            else:
                objective = await self._extract_objective(message)

            self.objective = objective
            self.start_time = time.time()
            self._runner = self._build_pipeline()

            logger.info(
                "objective=<%s> | starting execution",
                objective[:100],
            )
            span.set_attribute("workflow.objective", objective[:200])

            try:
                result = await self._execute_workflow(request_params, span)
                span.set_attribute("workflow.success", True)
                span.set_attribute("workflow.iterations", self.iteration)
                span.set_attribute("workflow.tokens_used", self.budget.tokens_used)
                span.set_attribute("workflow.cost", self.budget.cost_incurred)

                logger.info(
                    "iterations=<%d>, tokens=<%d>, cost=<$%.2f> | execution completed",
                    self.iteration,
                    self.budget.tokens_used,
                    self.budget.cost_incurred,
                )
                return result

            except Exception as e:
                span.set_attribute("workflow.success", False)
                span.record_exception(e)
                logger.error("error=<%s> | workflow failed", e, exc_info=True)
                return await self._emergency_completion(str(e))

    async def generate_str(
        self,
        message: str | MessageParamT | List[MessageParamT],
        request_params: RequestParams | None = None,
    ) -> str:
        """Generate and return string representation.

        Fixed: original returned ``str(messages[0])`` which stringified the
        message object instead of extracting text content.
        """
        messages = await self.generate(message, request_params)
        return _extract_text_from_messages(messages)

    async def generate_structured(
        self,
        message: str | MessageParamT | List[MessageParamT],
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Generate structured output."""
        result_str = await self.generate_str(message, request_params)

        parser = Agent(
            name="StructuredParser",
            instruction="Parse the content into the requested structure accurately.",
            context=self.context,
        )

        llm = self.llm_factory(parser)
        return await llm.generate_structured(
            message=f"<parse_request>\n{result_str}\n</parse_request>",
            response_model=response_model,
            request_params=RequestParams(max_iterations=1),
        )

    # ------------------------------------------------------------------
    # Core execution loop
    # ------------------------------------------------------------------

    async def _execute_workflow(
        self,
        request_params: Optional[RequestParams],
        span: "Span",
    ) -> List[MessageT]:
        """Aspect-wrapped plan → execute → verify → replan loop."""
        assert self._runner is not None

        # Shared state passed through all blocks via BlockContext
        state = {
            "objective": self.objective,
            "available_servers": self.available_servers,
            "artifacts": self.memory.artifacts,
            "all_knowledge": self.knowledge.items,
        }

        ctx = BlockContext(state=state, objective=self.objective)

        # --- Phase 1: Initial Planning ---
        logger.info("phase=<1> | creating initial plan")
        span.add_event("phase_1_planning")

        plan_result = await self._runner.run_block("planner", ctx)
        self._runner.apply_state_updates(ctx, plan_result)

        plan: Optional[Plan] = ctx.state.get("current_plan")

        if ctx.state.get("plan_is_complete"):
            logger.info("plan_is_complete=<true> | objective already satisfied")
            return await self._create_simple_response(
                plan.reasoning if plan else "Objective already satisfied."
            )

        if plan:
            self.current_plan = plan
            self.queue.load_plan(plan)

        # --- Phase 2: Iterative execute → verify → replan loop ---
        while self.iteration < self.config.execution.max_iterations:
            self.iteration += 1
            ctx.iteration = self.iteration

            logger.info(
                "iteration=<%d/%d> | main loop",
                self.iteration,
                self.config.execution.max_iterations,
            )
            span.add_event(
                "iteration",
                {"iteration": self.iteration},
            )

            # Check if queue is empty → verify completion
            if self.queue.is_empty():
                verify_result = await self._runner.run_block("verifier", ctx)
                self._runner.apply_state_updates(ctx, verify_result)

                if verify_result.routing == RoutingHint.FORCE_COMPLETE:
                    logger.info("verification=<complete> | objective verified")
                    break

                if verify_result.routing == RoutingHint.REPLAN:
                    # Replan: run planner again
                    ctx.state["progress_summary"] = self.queue.get_progress_summary()
                    ctx.state["knowledge_summary"] = self.knowledge.get_summary(
                        limit=15,
                    )
                    ctx.state["knowledge_items_for_planning"] = [
                        {
                            "key": item.key,
                            "value": item.value,
                            "confidence": item.confidence,
                            "category": item.category,
                        }
                        for item in self.knowledge.query(
                            self.objective,
                            limit=10,
                        )
                    ]
                    ctx.state["completed_step_descriptions"] = [
                        s.description for s in self.queue.completed_steps[-5:]
                    ]

                    replan_result = await self._runner.run_block("planner", ctx)
                    self._runner.apply_state_updates(ctx, replan_result)

                    new_plan = ctx.state.get("current_plan")
                    if new_plan and new_plan.steps:
                        added = self.queue.merge_plan(new_plan)
                        if added == 0:
                            logger.info("replan_added=<0> | no new steps, completing")
                            break
                        self.replan_count += 1
                        if self.replan_count >= self.config.execution.max_replans:
                            logger.warning(
                                "replan_count=<%d/%d> | max replans reached",
                                self.replan_count,
                                self.config.execution.max_replans,
                            )
                            break
                    else:
                        break

                    continue

                if verify_result.routing == RoutingHint.EMERGENCY_STOP:
                    logger.error("routing=<EMERGENCY_STOP> | aborting")
                    break

            # Get next step from queue
            next_step = self.queue.get_next_step()
            if not next_step:
                logger.info("queue=<empty> | no more steps")
                break

            # Execute step via ExecutorBlock
            ctx.state["current_step"] = next_step

            exec_result = await self._runner.run_block("executor", ctx)
            self._runner.apply_state_updates(ctx, exec_result)

            # Process results
            self.queue.complete_step(next_step)
            step_results = ctx.state.get("step_results", [])

            # Extract and store knowledge from task results
            for task_result in step_results:
                if task_result.success and task_result.output:
                    self.memory.add_task_result(task_result)
                    # Inline knowledge extraction (no separate LLM call per task)
                    self._extract_inline_knowledge(task_result)

            # Check routing hints from aspects
            if exec_result.routing == RoutingHint.FORCE_COMPLETE:
                logger.warning("routing=<FORCE_COMPLETE> | budget/policy limit")
                break
            if exec_result.routing == RoutingHint.EMERGENCY_STOP:
                logger.error("routing=<EMERGENCY_STOP> | critical failure")
                break

            # Context window management (configurable, not hardcoded 40000)
            context_size = self.memory.estimate_context_size()
            max_context = self.config.context.context_window_limit
            if context_size > max_context:
                logger.warning(
                    "context_size=<%d>, max=<%d> | trimming",
                    context_size,
                    max_context,
                )
                self.memory.trim_for_context(int(max_context * 0.75))

        # --- Phase 3: Final Synthesis ---
        span.add_event("phase_3_synthesis")
        logger.info("phase=<3> | creating final synthesis")

        # Prepare synthesis context
        ctx.state["completed_steps_data"] = self._build_completed_steps_data()
        ctx.state["all_knowledge"] = self.knowledge.items
        ctx.state["iteration_count"] = self.iteration
        ctx.state["tasks_completed_count"] = len(self.queue.completed_task_names)
        ctx.state["budget_info"] = {
            "tokens_used": self.budget.tokens_used,
            "cost_incurred": self.budget.cost_incurred,
        }

        synth_result = await self._runner.run_block("synthesizer", ctx)
        self._runner.apply_state_updates(ctx, synth_result)

        messages = ctx.state.get("synthesis_messages", [])
        if messages:
            return messages

        # Fallback: create response from synthesis output text
        output = ctx.state.get("synthesis_output", "")
        if output:
            return await self._create_simple_response(output)

        return await self._emergency_completion("Synthesis produced no output")

    # ------------------------------------------------------------------
    # Knowledge extraction (inline, no extra LLM call per task)
    # ------------------------------------------------------------------

    def _extract_inline_knowledge(self, task_result) -> None:
        """Extract knowledge from task output without an extra LLM call.

        Replaces the original KnowledgeExtractor which burned tokens
        on an LLM call after every single task.  This uses heuristic
        extraction for immediate knowledge capture.
        """
        output = task_result.output or ""
        if len(output) < 50:
            return

        # Extract key-value patterns from output
        # e.g. "Found that X is Y", "Discovered: Z", "Result: W"
        item = KnowledgeItem(
            key=f"Result from {task_result.task_name}",
            value=output[:500] if len(output) > 500 else output,
            source=task_result.task_name,
            confidence=0.7 if task_result.success else 0.3,
            category="task_output",
        )
        self.knowledge.add(item)

        # Also save as artifact if it looks like it contains structured data
        if any(
            phrase in output.lower()
            for phrase in ["created file:", "saved to:", "wrote to:"]
        ):
            self.memory.save_artifact(f"task_{task_result.task_name}_output", output)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_completed_steps_data(self) -> list:
        """Build completed steps data for synthesis context."""
        completed_steps = []
        for step in self.queue.completed_steps:
            step_data = {"description": step.description, "task_results": []}
            step_task_names = {t.name for t in step.tasks}
            step_results = [
                r for r in self.memory.task_results if r.task_name in step_task_names
            ]
            for result in step_results:
                if result.success and result.output:
                    task = self.queue.all_tasks.get(result.task_name)
                    task_desc = task.description if task else "Unknown task"
                    step_data["task_results"].append(
                        {
                            "description": task_desc,
                            "output": result.output,
                            "success": True,
                        }
                    )
            completed_steps.append(step_data)
        return completed_steps

    async def _emergency_completion(self, error: str) -> List[MessageT]:
        """Provide best-effort response when workflow fails."""
        logger.warning("error=<%s> | entering emergency completion", error)

        emergency_agent = Agent(
            name="EmergencyResponder",
            instruction=EMERGENCY_RESPONDER_INSTRUCTION,
            context=self.context,
        )

        partial_knowledge = [
            {"key": item.key, "value": item.value} for item in self.knowledge.items[:10]
        ]
        artifacts_created = (
            list(self.memory.artifacts.keys())[:5] if self.memory.artifacts else None
        )

        context = get_emergency_context(
            objective=self.objective,
            error=error,
            progress_summary=self.queue.get_progress_summary(),
            partial_knowledge=partial_knowledge,
            artifacts_created=artifacts_created,
        )
        prompt = get_emergency_prompt(context)

        async with emergency_agent:
            llm = await emergency_agent.attach_llm(self.llm_factory)
            return await llm.generate(message=prompt)

    async def _extract_objective(
        self, message: MessageParamT | List[MessageParamT]
    ) -> str:
        """Extract objective from complex message types."""
        extractor = Agent(
            name="ObjectiveExtractor",
            instruction=(
                "Extract the user's objective or request from their message. "
                "Be concise and clear."
            ),
            context=self.context,
        )
        llm = self.llm_factory(extractor)
        return await llm.generate_str(
            message=message,
            request_params=RequestParams(max_iterations=1),
        )

    async def _create_simple_response(self, content: str) -> List[MessageT]:
        """Create a simple response message."""
        simple_agent = Agent(
            name="SimpleResponder",
            instruction="Provide a clear, direct response.",
            context=self.context,
        )
        async with simple_agent:
            llm = await simple_agent.attach_llm(self.llm_factory)
            return await llm.generate(message=content)

    def get_health_summary(self) -> dict:
        """Get pipeline health summary (exposed for monitoring)."""
        if self._health_gate:
            return self._health_gate.summary()
        return {}

    def get_knowledge_stats(self) -> dict:
        """Get knowledge store statistics (exposed for monitoring)."""
        return self.knowledge.get_stats()


# ---------------------------------------------------------------------------
# Message content extraction (fixes broken generate_str)
# ---------------------------------------------------------------------------


def _extract_text_from_messages(messages: list) -> str:
    """Extract text content from message objects.

    Fixes the original broken ``generate_str`` which did
    ``str(messages[0])`` and got a repr string instead of content.
    """
    if not messages:
        return ""

    parts = []
    for msg in messages:
        if hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
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

    return "\n".join(parts) if parts else ""
