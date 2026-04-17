"""Aspect-oriented pipeline block framework.

Ported from MiroThinker's ``models/pipeline_block.py``.  Every orchestrator
phase is a fenced ``PipelineBlock`` with declared inputs, outputs, and typed
validation rules.  Cross-cutting concerns (health gates, budget tracking,
error escalation, timing) are applied uniformly as ``Aspect`` instances by
the ``PipelineRunner``.

Key separation of concerns:

- **Blocks** own business logic + validation RULES (per data type).
- **Aspects** own cross-cutting CONSEQUENCES -- what happens when rules
  fail, when errors occur, when phases start/end.

Architecture::

    PipelineRunner (stateful aspect-application engine)
    +-- aspects: list[Aspect]       <-- applied to EVERY block
    +-- registered blocks
          +-- PlannerBlock / ExecutorBlock / VerifierBlock / SynthesizerBlock

    The orchestrator's while-loop manages sequencing.
    Each iteration delegates to runner.run_block() for the active phase.

    For each block:
      1. aspect.before(block, ctx) -- first non-None SHORT-CIRCUITS
      2. block.execute(ctx)        -- business logic only
      3. aspect.after(block, ctx, result) -- reversed order
      on error: aspect.on_error(block, ctx, error) -- reversed order

Blocks NEVER touch health tracking, budget management, dashboard events,
or error handling directly.  Those are aspect responsibilities.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class RoutingHint(str, Enum):
    """What a block tells the runner to do next."""

    CONTINUE = "CONTINUE"
    REPLAN = "REPLAN"
    FORCE_COMPLETE = "FORCE_COMPLETE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class BlockCriticality(str, Enum):
    """How critical a block is -- determines error escalation policy.

    The ErrorEscalationAspect uses this to decide whether to absorb
    or propagate errors.
    """

    CRITICAL = "CRITICAL"
    BEST_EFFORT = "BEST_EFFORT"


# ---------------------------------------------------------------------------
# I/O Specifications -- functional-programming style parameter contracts
# ---------------------------------------------------------------------------


@dataclass
class ParamSpec:
    """Typed parameter specification for a block's input or output.

    The RULES live here (owned by the block).  The CONSEQUENCES of rule
    failure live in aspects (cross-cutting).
    """

    key: str
    expected_type: type = str
    validator: Optional[Callable[[Any], bool]] = None
    description: str = ""
    required: bool = True
    default: Any = None

    def validate(self, value: Any) -> tuple[bool, str]:
        """Validate a value against this spec.

        Returns (ok, error_message).  Empty error_message when ok=True.
        """
        if value is None and not self.required:
            return True, ""

        if value is None and self.required:
            return False, f"Required key '{self.key}' is missing"

        if not isinstance(value, self.expected_type):
            return False, (
                f"Key '{self.key}' has type {type(value).__name__}, "
                f"expected {self.expected_type.__name__}"
            )

        if self.validator is not None:
            try:
                ok = self.validator(value)
            except Exception as exc:
                return False, f"Validator for '{self.key}' raised: {exc}"
            if not ok:
                desc = self.description or f"validator for '{self.key}'"
                return False, f"Value for '{self.key}' failed validation: {desc}"

        return True, ""


# ---------------------------------------------------------------------------
# Block Context -- injected into every block's execute()
# ---------------------------------------------------------------------------


@dataclass
class BlockContext:
    """Everything a block needs to do its work.

    Injected by the PipelineRunner.  Blocks NEVER reach into module
    globals -- they receive all dependencies through this context.
    """

    state: Dict[str, Any]
    objective: str = ""
    iteration: int = 0
    # Aspect-managed metadata (aspects read/write these, blocks don't)
    _phase_start_time: float = 0.0
    _cost_snapshot: float = 0.0


# ---------------------------------------------------------------------------
# Block Result -- returned by every block's execute()
# ---------------------------------------------------------------------------


@dataclass
class BlockResult:
    """Structured result from a block's execution.

    The block returns metrics and state updates; aspects handle health
    tracking, dashboard events, and error escalation based on these.
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    routing: RoutingHint = RoutingHint.CONTINUE
    diagnosis: str = ""
    output: Optional[str] = None


# ---------------------------------------------------------------------------
# PipelineBlock -- the fenced phase abstraction
# ---------------------------------------------------------------------------


class PipelineBlock(ABC):
    """A single fenced phase in the orchestrator pipeline.

    The block ONLY contains business logic and declares its own
    validation rules via ``input_specs`` / ``output_specs``.
    Cross-cutting consequences are handled by aspects.
    """

    name: str = ""
    input_specs: List[ParamSpec] = []
    output_specs: List[ParamSpec] = []
    criticality: BlockCriticality = BlockCriticality.BEST_EFFORT

    @abstractmethod
    async def execute(self, ctx: BlockContext) -> BlockResult:
        """Execute the block's business logic."""
        ...


# ---------------------------------------------------------------------------
# Aspect -- cross-cutting concern applied to every block
# ---------------------------------------------------------------------------


class Aspect(ABC):
    """A cross-cutting concern applied uniformly to every pipeline block.

    Aligned with MiroThinker's Aspect pattern:
    - ``before()`` returns ``Optional[BlockResult]``.
      First non-None return **short-circuits** (block won't execute).
    - ``after()`` may modify the result.
    - ``on_error()`` may return an override BlockResult.
    """

    name: str = ""

    async def before(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
    ) -> Optional[BlockResult]:
        """Return None to continue, or a BlockResult to short-circuit."""
        return None

    async def after(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        """Called after block.execute() succeeds.  May modify result."""

    async def on_error(
        self,
        block: PipelineBlock,
        ctx: BlockContext,
        error: Exception,
    ) -> Optional[BlockResult]:
        """Return None to let runner handle it, or a BlockResult to override."""
        return None


# ---------------------------------------------------------------------------
# PipelineRunner -- stateful aspect-application engine
# ---------------------------------------------------------------------------


class PipelineRunner:
    """Applies aspects uniformly around each pipeline block's execution.

    The runner is NOT a sequencer -- the orchestrator's while-loop manages
    the flow.  The runner provides ``run_block()`` which the orchestrator
    invokes for each phase.
    """

    def __init__(
        self,
        blocks: List[PipelineBlock],
        aspects: List[Aspect],
    ) -> None:
        self.blocks = blocks
        self.aspects = aspects
        self._block_map: Dict[str, PipelineBlock] = {b.name: b for b in blocks}
        self.consecutive_failures: int = 0

    def get_block(self, name: str) -> Optional[PipelineBlock]:
        """Look up a registered block by name."""
        return self._block_map.get(name)

    async def run_block(
        self,
        block_name: str,
        ctx: BlockContext,
    ) -> BlockResult:
        """Run a named block with all aspects applied.

        Aspect execution order:
          before: aspects[0] .. aspects[N] (first non-None short-circuits)
          after:  aspects[N] .. aspects[0]
          error:  aspects[N] .. aspects[0]
        """
        block = self._block_map.get(block_name)
        if block is None:
            logger.error("Unknown block: '%s'", block_name)
            return BlockResult(
                metrics={"error": f"Unknown block: {block_name}"},
                routing=RoutingHint.EMERGENCY_STOP,
            )

        # --- before (in order) ---
        ran_before: List[Aspect] = []
        for aspect in self.aspects:
            try:
                short_circuit = await aspect.before(block, ctx)
                ran_before.append(aspect)
                if short_circuit is not None:
                    logger.info(
                        "Aspect '%s' short-circuited block '%s'",
                        aspect.name,
                        block.name,
                    )
                    # Run after() only for aspects whose before() already ran
                    for done in reversed(ran_before):
                        try:
                            await done.after(block, ctx, short_circuit)
                        except Exception as after_exc:
                            logger.warning(
                                "Aspect '%s' after() during short-circuit: %s",
                                done.name,
                                after_exc,
                            )
                    return short_circuit
            except Exception as exc:
                ran_before.append(aspect)
                logger.warning(
                    "Aspect '%s' before() failed for '%s': %s",
                    aspect.name,
                    block.name,
                    exc,
                )

        # --- execute ---
        result: BlockResult
        try:
            result = await block.execute(ctx)
            self.consecutive_failures = 0
        except Exception as exc:
            self.consecutive_failures += 1

            # --- on_error (reversed) ---
            override: Optional[BlockResult] = None
            for aspect in reversed(self.aspects):
                try:
                    aspect_result = await aspect.on_error(block, ctx, exc)
                    if aspect_result is not None and override is None:
                        override = aspect_result
                except Exception as aspect_exc:
                    logger.warning(
                        "Aspect '%s' on_error() failed for '%s': %s",
                        aspect.name,
                        block.name,
                        aspect_exc,
                    )

            if override is not None:
                result = override
            else:
                result = BlockResult(
                    metrics={"error": str(exc), "block_failed": True},
                    routing=RoutingHint.CONTINUE,
                    diagnosis=f"Block '{block.name}' failed: {exc}",
                )

            # after() even on error (cleanup)
            for aspect in reversed(self.aspects):
                try:
                    await aspect.after(block, ctx, result)
                except Exception as aspect_exc:
                    logger.warning(
                        "Aspect '%s' after() post-error: %s",
                        aspect.name,
                        aspect_exc,
                    )
            return result

        # --- after (reversed) ---
        for aspect in reversed(self.aspects):
            try:
                await aspect.after(block, ctx, result)
            except Exception as exc:
                logger.warning(
                    "Aspect '%s' after() failed for '%s': %s",
                    aspect.name,
                    block.name,
                    exc,
                )

        return result

    def apply_state_updates(
        self,
        ctx: BlockContext,
        result: BlockResult,
    ) -> None:
        """Write a block's declared state updates back to the context state."""
        for key, value in result.state_updates.items():
            ctx.state[key] = value
