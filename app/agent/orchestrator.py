"""Deterministic agent orchestrator for the AI WhatsApp Support Agent.

``AgentOrchestrator.handle_turn`` coordinates one customer turn across the
pieces built in earlier Milestone 2 slices:

    ConversationStore  -> load / save ``ConversationState``
    guardrails         -> deterministic detectors + ``GroundingValidator``
    EscalationPolicy   -> one ``EscalationDecision`` per evaluation
    LeadExtractor      -> validated ``LeadDelta`` from the customer message
    PromptBuilder      -> business-aware messages for the LLM
    LLMProvider        -> ``complete(messages, tools=...)`` (native tool calling)
    ToolRegistry       -> tool specs + deterministic tool execution

Architectural principle: the LLM is NOT the workflow controller. Python owns
state, validation, guardrails, escalation, tool execution, tool-call limits,
error handling, persistence and the turn result. The LLM only supplies
natural language and tool-call requests.

Design constraints (Milestone 2, Slice 6):
- Not wired into ``app.main`` or the webhook. The Milestone 1 request path
  is untouched; switching the webhook over is a later slice.
- Explicit bounded loops: at most ``AGENT_MAX_TOOL_ROUNDS`` tool rounds per
  turn, then ONE final text-only call with ``tools`` omitted entirely
  (never ``tool_choice="none"`` — confirmed unreliable on Groq in Slice 0).
- Tool calls are executed sequentially, in the order the model returned
  them; gpt-oss-120b on Groq does not reliably emit parallel calls.
- ``handle_turn`` never raises for provider or tool failures: it logs the
  exception type only, returns a deterministic safe fallback reply, and
  still persists valid state.

Lead extraction (Milestone 2, Slice 8):
- An optional ``LeadExtractor`` runs exactly once per customer turn, after
  the incoming guardrails allow the turn to continue and before the first
  prompt is built, so the model sees the current lead state. It never runs
  inside the tool loop and never runs on a turn the policy short-circuits.
- The extractor stays pure: its ``LeadDelta`` enters state only through
  ``ConversationState.apply_lead_delta``, which owns merge/provenance and
  recomputes qualification deterministically. The model never sets
  qualification or escalation; ``LeadDelta`` has no such fields.
- Extraction is enrichment, not the response path: a failed, malformed,
  or crashing extraction leaves the lead profile untouched and the turn
  continues normally. It never triggers the safe fallback by itself.

Guardrails + escalation (Milestone 2, Slice 10):
- Incoming detectors (``InjectionDetector``, ``AngerScorer``,
  ``RepetitionDetector``, ``HumanRequestDetector``) run exactly once per
  customer turn, on the customer text only — never on tool output or
  model output. They are pure; the orchestrator owns every state change.
- ``EscalationPolicy`` is evaluated on the incoming signals against the
  state *before* those signals are applied to ``ConversationFlags`` (the
  Slice 9 call-order contract: the policy adds this turn's contribution
  itself). Only afterwards does ``apply_signals_to_flags`` run, once.
- A blocking incoming decision (``escalate`` / ``refuse`` / ``clarify``)
  is answered with a deterministic template from this module. No LLM call
  and no lead extraction happen on such a turn.
- Every candidate model reply is checked by ``GroundingValidator`` against
  the current-turn tool results *and* the ``KnowledgeBase`` before it may
  be sent. A rejected reply is never sent; the policy is re-evaluated with
  a grounding-only signal bundle (so no incoming counter is double
  counted), at most ONE tool-free corrective generation is attempted, and
  if that is also rejected a deterministic recovery reply goes out.
- ``handoff_ready`` is recorded, not enforced: a qualified, complete lead
  keeps receiving grounded model replies. The explicit
  ``mark_handoff_ready`` transition is a consent step owned by a later
  qualification-policy slice; Python state stays authoritative either way.
- Fail closed: a detector error skips the LLM for the turn (safe fallback),
  a policy error escalates, a validator error counts as ungrounded.

Human handoff (Milestone 2, Slice 12):
- An optional ``HandoffSink`` (``app.agent.handoff``) is injected through
  the constructor; there is no module-level sink. ``EscalationPolicy``
  stays the decision maker and the sink stays the delivery/storage
  boundary; the orchestrator only carries the *applied* decision across.
- Exactly one submission point per turn: after the reply is final and the
  turn's messages are in ``history`` (so the human sees the triggering
  message and our reply), before the state is saved. The request is built
  only by ``handoff_request_from_decision``; it returns ``None`` for every
  action that does not ask for a human, so no second action table lives
  here. ``escalate`` yields an ``escalation`` handoff, ``handoff_ready`` a
  ``qualified_lead`` handoff; the model reply for ``handoff_ready`` still
  goes out and qualification is not transitioned (Slice 10 semantics).
- Deduplication belongs to the sink alone. The policy keeps returning
  ``escalate`` for the life of an escalation; each such turn is submitted
  once and the sink reports ``deduplicated``.
- A sink that raises, rejects, or is absent never fails the turn: the
  deterministic handoff reply still goes out, the existing state
  transition (``mark_escalated``) still happens, diagnostics record the
  outcome (``created``/``deduplicated``/``rejected``/``unavailable``) and
  the exception type only. No retries within a turn.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agent.escalation import EscalationAction, EscalationDecision, EscalationPolicy, UserMessageInstruction
from app.agent.extraction import ExtractionResult, LeadExtractor
from app.agent.guardrails import (
    AngerScorer,
    GroundingResult,
    GroundingValidator,
    GuardrailSignals,
    HumanRequestDetector,
    InjectionDetector,
    RepetitionDetector,
    apply_signals_to_flags,
)
from app.agent.handoff import HandoffResult, HandoffSink, handoff_request_from_decision
from app.agent.prompts import PromptBuilder
from app.agent.state import ConversationState, EscalationStatus, MAX_MESSAGE_LENGTH, ToolInvocation
from app.agent.store import ConversationStore
from app.config import mask_phone_number
from app.knowledge import KnowledgeBase
from app.llm.base import ChatMessage, LLMProvider, LLMResponse, ToolCall
from app.llm.base import ToolSpec as LLMToolSpec
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds and constants
# ---------------------------------------------------------------------------

AGENT_MAX_TOOL_ROUNDS = 2
# After a tool reports ``invalid_input`` the model may correct itself once.
MAX_INVALID_INPUT_RETRIES = 1
# After the grounding validator rejects a reply the model may rewrite it once
# (text-only, no tools). Never more: regeneration is bounded by construction.
MAX_CORRECTIVE_GENERATIONS = 1

SAFE_FALLBACK_REPLY = "Sorry — I’m having trouble checking that right now. Let me get the team to help."

# Deterministic customer-facing replies the policy can require. The policy
# modules only emit codes; this is the single place those codes become text.
# The model never writes a refusal, handoff, or clarification on the
# policy's behalf.
SAFE_REFUSAL_REPLY = (
    "I can help with the coffee, orders, and support questions, "
    "but I can't provide private system or credential information."
)
HUMAN_HANDOFF_REPLY = "I'll get a member of the team to help with this."
CLARIFICATION_REPLY = "I want to make sure I understand. Could you clarify what you need?"
UNVERIFIED_RECOVERY_REPLY = "Sorry — I couldn't verify that information reliably. Let me get the team to help."

_INSTRUCTION_REPLIES: Dict[UserMessageInstruction, str] = {
    UserMessageInstruction.PROVIDE_SAFE_REFUSAL: SAFE_REFUSAL_REPLY,
    UserMessageInstruction.OFFER_HUMAN_HANDOFF: HUMAN_HANDOFF_REPLY,
    UserMessageInstruction.ASK_FOR_CLARIFICATION: CLARIFICATION_REPLY,
    UserMessageInstruction.SUPPRESS_UNGROUNDED_CLAIM: UNVERIFIED_RECOVERY_REPLY,
}

# Instruction appended (as a system message) for the single corrective
# generation. Carries grounding reason *codes* only — never customer text.
CORRECTIVE_INSTRUCTION = (
    "Your previous draft reply (the assistant message just above) stated product details that are not "
    "supported by this turn's tool results or the business knowledge in your instructions "
    "(problems: {codes}). Rewrite the reply using only facts that appear in those sources. Do not state "
    "any price, availability, origin, or tasting note you cannot see there; if a detail cannot be "
    "verified, say you will check with the team instead. Do not mention this correction."
)

# Actions that end the turn with a deterministic reply and no model call.
_BLOCKING_ACTIONS = frozenset(
    {EscalationAction.ESCALATE, EscalationAction.REFUSE, EscalationAction.CLARIFY, EscalationAction.SUPPRESS}
)

# Fail-closed verdict used when the policy itself raises: route to a human.
_POLICY_ERROR_DECISION = EscalationDecision(
    action=EscalationAction.ESCALATE,
    reason_codes=["policy_error"],
    user_message_instruction=UserMessageInstruction.OFFER_HUMAN_HANDOFF,
    priority=1,
)

_ESCALATION_REASON_MAX_LENGTH = 200  # matches EscalationState.reason

# Tool-result statuses shared with ``app.tools`` (see ``ProductLookupStatus``).
STATUS_OK = "ok"
STATUS_NO_MATCH = "no_match"
STATUS_INVALID_INPUT = "invalid_input"
STATUS_UNAVAILABLE = "unavailable"

_TOOL_NAME_MAX_LENGTH = 64  # matches ToolInvocation.tool_name
_RAW_ARGUMENTS_PREVIEW_LENGTH = 200

ToolCallDisposition = Literal[
    "executed",
    "rejected_unknown_tool",
    "rejected_malformed_arguments",
    "rejected_retry_budget",
    "rejected_tool_unavailable",
]

LoopExit = Literal[
    "text_reply",
    "tool_round_limit",
    "tool_unavailable",
    "invalid_input_retries_exhausted",
    "empty_reply",
    "llm_error",
    "not_run",  # the policy or a guardrail failure ended the turn before any model call
]

FallbackReason = Literal["empty_reply", "llm_error", "guardrail_error", "ungrounded_reply"]

ReplySource = Literal["model", "model_corrected", "policy", "fallback"]
EscalationStage = Literal["incoming", "outgoing"]

ExtractionSource = Literal["model", "model_repaired", "fallback"]
_EXTRACTION_SOURCES = ("model", "model_repaired", "fallback")

# The sink's own ``HandoffOutcome`` values plus ``unavailable``: no sink is
# configured, the sink raised, or it returned something unusable.
HandoffOutcomeCode = Literal["created", "deduplicated", "rejected", "unavailable"]


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """One tool call requested by the model and what the orchestrator did with it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str
    tool_name: str
    round: int = Field(..., ge=1)
    disposition: ToolCallDisposition
    status: str = Field(..., description="Result status fed back to the model (ok/no_match/invalid_input/unavailable)")
    ok: bool
    arguments: Dict[str, Any] = Field(default_factory=dict)
    raw_arguments_preview: str = ""
    error_code: Optional[str] = None


class TurnDiagnostics(BaseModel):
    """Safe, structured facts about how a turn ran.

    Never contains prompt text, customer text, model output, credentials,
    authorization headers, or an unmasked phone number. Guardrail fields
    carry scores, counts and codes only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sender: str = Field(..., description="Masked sender ID")
    message_id: Optional[str] = None
    turn: int = Field(..., ge=0)
    llm_calls: int = Field(0, ge=0)
    tool_rounds: int = Field(0, ge=0)
    tool_calls_requested: int = Field(0, ge=0)
    tool_calls_executed: int = Field(0, ge=0)
    tool_calls_rejected: int = Field(0, ge=0)
    tool_round_limit_reached: bool = False
    final_call_tools_omitted: bool = False
    loop_exit: LoopExit
    fallback_used: bool = False
    fallback_reason: Optional[FallbackReason] = None
    error_type: Optional[str] = Field(None, description="Exception class name only, never its message")
    usage_prompt_tokens: int = Field(0, ge=0)
    usage_completion_tokens: int = Field(0, ge=0)
    # Lead extraction (field NAMES only; never extracted values or raw model output).
    extraction_attempted: bool = False
    extraction_success: bool = False
    extraction_source: Optional[ExtractionSource] = None
    extracted_field_names: List[str] = Field(default_factory=list)
    extraction_errors_count: int = Field(0, ge=0)
    extraction_error_type: Optional[str] = Field(None, description="Exception class name only, never its message")
    # Incoming guardrails (pattern counts and scores only).
    injection_suspected: bool = False
    injection_hit_count: int = Field(0, ge=0)
    anger_score: float = Field(0.0, ge=0.0, le=1.0)
    repetition_detected: bool = False
    human_requested: bool = False
    guardrail_error_types: List[str] = Field(default_factory=list, description="Exception class names only")
    # Outgoing grounding.
    grounding_checks: int = Field(0, ge=0, description="Candidate replies run through the validator")
    grounding_violation_count: int = Field(0, ge=0, description="Candidate replies the validator rejected")
    grounding_reason_codes: List[str] = Field(default_factory=list)
    grounding_error_type: Optional[str] = Field(None, description="Exception class name only, never its message")
    corrective_generation_attempted: bool = False
    # Escalation policy: the decisive decision for the turn.
    escalation_action: Optional[str] = None
    escalation_reason_codes: List[str] = Field(default_factory=list)
    escalation_stage: Optional[EscalationStage] = None
    policy_error_type: Optional[str] = Field(None, description="Exception class name only, never its message")
    reply_source: ReplySource = "model"
    # Human handoff (outcome codes and the sink's opaque ID only; never the
    # request payload, which carries the transcript and lead details).
    handoff_attempted: bool = Field(False, description="A request was submitted to a configured sink")
    handoff_accepted: bool = False
    handoff_outcome: Optional[HandoffOutcomeCode] = None
    handoff_id: Optional[str] = Field(None, description="Sink-assigned correlation ID; never shown to the customer")
    handoff_kind: Optional[str] = None
    handoff_priority: Optional[str] = None
    handoff_error_type: Optional[str] = Field(None, description="Exception class name only, never its message")


class AgentTurnResult(BaseModel):
    """The structured outcome of one ``handle_turn`` call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reply_text: str = Field(..., min_length=1)
    state_snapshot: ConversationState
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    diagnostics: TurnDiagnostics


# ---------------------------------------------------------------------------
# Per-turn scratch
# ---------------------------------------------------------------------------


@dataclass
class _TurnContext:
    """Mutable bookkeeping for exactly one turn. Never shared or stored."""

    state: ConversationState
    text: str
    # Messages appended after the current customer message: tool-protocol
    # exchanges (assistant(tool_calls) + one role="tool" per call) and, for
    # a corrective generation, the rejected draft plus its instruction.
    transcript: List[ChatMessage] = field(default_factory=list)
    records: List[ToolCallRecord] = field(default_factory=list)
    llm_calls: int = 0
    tool_rounds: int = 0
    invalid_input_failures: int = 0
    # Once set, no further tool calls are executed this turn and the loop
    # proceeds straight to the final text-only call.
    tools_closed: Optional[Literal["tool_unavailable", "invalid_input_retries_exhausted"]] = None
    tool_round_limit_reached: bool = False
    final_call_tools_omitted: bool = False
    usage_prompt_tokens: int = 0
    usage_completion_tokens: int = 0
    # Lead extraction outcome (set at most once per turn, before the LLM loop).
    extraction_attempted: bool = False
    extraction_success: bool = False
    extraction_source: Optional[ExtractionSource] = None
    extracted_field_names: List[str] = field(default_factory=list)
    extraction_errors_count: int = 0
    extraction_error_type: Optional[str] = None
    # Incoming guardrails (run exactly once per turn).
    signals: GuardrailSignals = field(default_factory=GuardrailSignals)
    guardrail_error_types: List[str] = field(default_factory=list)
    # Outgoing grounding.
    grounding_checks: int = 0
    grounding_violation_count: int = 0
    grounding_reason_codes: List[str] = field(default_factory=list)
    grounding_error_type: Optional[str] = None
    corrective_generation_attempted: bool = False
    # Escalation policy.
    decision: Optional[EscalationDecision] = None
    decision_stage: Optional[EscalationStage] = None
    policy_error_type: Optional[str] = None
    reply_source: ReplySource = "model"
    fallback_reason: Optional[FallbackReason] = None
    # Human handoff. ``applied_decision`` is the one decision this turn acted
    # on (set by ``_apply_decision``); it is handed to the adapter exactly
    # once, after the reply is final. The adapter decides whether it warrants
    # a handoff at all.
    applied_decision: Optional[EscalationDecision] = None
    handoff_attempted: bool = False
    handoff_accepted: bool = False
    handoff_outcome: Optional[HandoffOutcomeCode] = None
    handoff_id: Optional[str] = None
    handoff_kind: Optional[str] = None
    handoff_priority: Optional[str] = None
    handoff_error_type: Optional[str] = None

    @property
    def invalid_input_budget_exhausted(self) -> bool:
        return self.invalid_input_failures > MAX_INVALID_INPUT_RETRIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(status: str, code: str, message: str) -> Dict[str, Any]:
    """A structured tool result the orchestrator produces without executing a tool.

    Mirrors the ``{"status", "error": {"code", "message", "fields"}}`` shape
    every tool handler uses, so the model sees one consistent contract.
    """
    return {
        "status": status,
        "error": {"code": code, "message": message, "fields": []},
    }


def _safe_tool_name(name: Any) -> str:
    """Bound a model-supplied tool name so it always fits ``ToolInvocation``."""
    text = str(name).strip() if name is not None else ""
    return text[:_TOOL_NAME_MAX_LENGTH] or "unknown"


def _preview(raw_arguments: Any) -> str:
    text = raw_arguments if isinstance(raw_arguments, str) else ""
    return text[:_RAW_ARGUMENTS_PREVIEW_LENGTH]


def _assistant_tool_call_message(response: LLMResponse, call_ids: List[str]) -> ChatMessage:
    """The assistant message that requested tools, in provider wire format."""
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": tc.name, "arguments": tc.raw_arguments},
        }
        for call_id, tc in zip(call_ids, response.tool_calls)
    ]
    return ChatMessage(role="assistant", content=response.content or "", tool_calls=tool_calls)


def _tool_result_message(call_id: str, result: Dict[str, Any]) -> ChatMessage:
    return ChatMessage(role="tool", content=json.dumps(result, default=str), tool_call_id=call_id)


def _result_status(result: Dict[str, Any]) -> str:
    status = result.get("status")
    return status if isinstance(status, str) and status else STATUS_UNAVAILABLE


def _result_error_code(result: Dict[str, Any]) -> Optional[str]:
    error = result.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return code if isinstance(code, str) else None
    return None


def _deterministic_reply(decision: EscalationDecision) -> str:
    """Map a policy instruction to its fixed customer-facing text.

    Unknown instructions fail closed to the handoff text rather than to a
    model-written reply.
    """
    return _INSTRUCTION_REPLIES.get(decision.user_message_instruction, HUMAN_HANDOFF_REPLY)


def _escalation_reason(decision: EscalationDecision) -> str:
    return ",".join(decision.reason_codes)[:_ESCALATION_REASON_MAX_LENGTH] or decision.action.value


def _ungrounded_by_error() -> GroundingResult:
    """The verdict used when the validator itself fails: unverified is unsafe."""
    return GroundingResult(grounded=False, violations=["validator_error"], reason_codes=["validator_error"])


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Coordinates one conversation turn: state -> guardrails -> prompt -> LLM -> tools -> grounding -> reply.

    The orchestrator is a coordinator, not a store: all history lives in
    ``ConversationState`` (persisted through ``ConversationStore``), all
    prompt text comes from ``PromptBuilder``, lead data comes only from the
    optional ``LeadExtractor`` (merged through ``ConversationState``), and
    every safety verdict comes from the pure detectors and
    ``EscalationPolicy``; this class only applies them. When the policy asks
    for a human, the optional ``HandoffSink`` receives the request; the
    orchestrator never decides that on its own.
    """

    def __init__(
        self,
        llm: LLMProvider,
        knowledge: KnowledgeBase,
        tools: ToolRegistry,
        store: ConversationStore,
        max_tool_rounds: int = AGENT_MAX_TOOL_ROUNDS,
        extractor: Optional[LeadExtractor] = None,
        policy: Optional[EscalationPolicy] = None,
        grounding: Optional[GroundingValidator] = None,
        injection: Optional[InjectionDetector] = None,
        anger: Optional[AngerScorer] = None,
        repetition: Optional[RepetitionDetector] = None,
        human_request: Optional[HumanRequestDetector] = None,
        handoff_sink: Optional[HandoffSink] = None,
    ):
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must not be negative")
        self._llm = llm
        self._knowledge = knowledge
        self._tools = tools
        self._store = store
        self._max_tool_rounds = max_tool_rounds
        # ``None`` disables lead extraction; the turn then runs exactly as in Slice 6.
        self._extractor = extractor
        # ``None`` means escalations are recorded in state only and reported
        # as ``unavailable``; the caller owns the sink, this module holds no
        # default instance.
        self._handoff_sink = handoff_sink
        # Guardrails and policy are always on; the arguments exist so tests
        # can substitute instrumented or failing components.
        self._policy = policy if policy is not None else EscalationPolicy()
        # The catalog is the trusted source of truth for product facts, so a
        # correct claim ("Yirgacheffe Light is ₹780.") grounds against it even
        # when product_lookup was not called; wrong or unknown claims still fail.
        self._grounding = grounding if grounding is not None else GroundingValidator(catalog_as_facts=True)
        self._injection = injection if injection is not None else InjectionDetector()
        self._anger = anger if anger is not None else AngerScorer()
        self._repetition = repetition if repetition is not None else RepetitionDetector()
        self._human_request = human_request if human_request is not None else HumanRequestDetector()

    # -- Public API ---------------------------------------------------------

    async def handle_turn(
        self,
        sender_id: str,
        text: str,
        message_id: Optional[str] = None,
    ) -> AgentTurnResult:
        """Run one customer turn end to end and return a structured result.

        Never raises for guardrail, policy, LLM or tool failures. Raises
        ``ValueError`` only for caller misuse (blank ``text``), before any
        state is touched.
        """
        customer_text = (text or "").strip()
        if not customer_text:
            raise ValueError("text must not be blank")
        customer_text = customer_text[:MAX_MESSAGE_LENGTH]

        # 1. Load state (a copy; nothing persists until save).
        state = self._store.get(sender_id)
        if state is None:
            state = ConversationState.new(sender_id)

        # 2. Begin turn: bumps turn_count and resets current-turn tool results.
        # The customer message is appended to history further down, once the
        # reply is final: PromptBuilder renders the current message itself,
        # delimited, as the final user message, so adding it to ``history``
        # first would send it twice (and the repetition detector compares
        # against *previous* customer turns only).
        turn = state.begin_turn()

        ctx = _TurnContext(state=state, text=customer_text)
        masked_sender = mask_phone_number(sender_id)

        # 3. Incoming guardrails: pure detectors, once, on the customer text only.
        self._analyze_incoming(ctx, masked_sender, turn)

        # 5. Policy on the incoming signals, evaluated BEFORE the flags absorb
        #    them (the policy adds this turn's contribution itself).
        decision = self._evaluate_policy(ctx, ctx.signals, stage="incoming")

        # 4. The orchestrator owns the flag mutation: exactly once for the
        #    incoming signals. Outgoing grounding bumps its own counter later
        #    through a grounding-only bundle, so nothing is counted twice.
        state.flags = apply_signals_to_flags(state.flags, ctx.signals)

        error_type: Optional[str] = None
        loop_exit: LoopExit = "not_run"
        if ctx.guardrail_error_types:
            # Fail closed: an unscreened message never reaches the model.
            reply_text = SAFE_FALLBACK_REPLY
            ctx.reply_source = "fallback"
            ctx.fallback_reason = "guardrail_error"
        elif decision.action in _BLOCKING_ACTIONS:
            # Deterministic reply; no extraction, no model, no tools.
            reply_text = _deterministic_reply(decision)
            ctx.reply_source = "policy"
            self._apply_decision(ctx, decision)
        else:
            # 6-7. Extract lead data once, merge it, recompute qualification.
            await self._update_lead(ctx, masked_sender, turn)

            # 8-12. Prompt -> LLM -> tools -> grounding -> policy -> reply.
            try:
                reply_text, loop_exit = await self._generate_reply(ctx)
            except Exception as exc:  # provider/tool failures must never escape
                error_type = type(exc).__name__
                logger.error(
                    "Agent turn failed for %s (turn %d): %s; returning safe fallback",
                    masked_sender,
                    turn,
                    error_type,
                )
                logger.debug("Agent turn failure detail", exc_info=exc)
                reply_text = SAFE_FALLBACK_REPLY
                ctx.reply_source = "fallback"
                ctx.fallback_reason = "llm_error"
                loop_exit = "llm_error"

        # 13. This turn's messages go into history first so a handoff carries
        #     the triggering message and our reply in its transcript.
        state.add_user_message(customer_text)
        state.add_assistant_message(reply_text)

        # 14. Human handoff for the decision this turn applied, if it asked
        #     for one. Exactly one submission per turn; never raises.
        self._submit_handoff(ctx, masked_sender, turn)

        # 15. Persist: messages, tool history, flags, escalation and lead state.
        self._store.save(state)

        diagnostics = self._diagnostics(ctx, masked_sender, message_id, turn, loop_exit, error_type)
        logger.info(
            "Agent turn complete for %s: turn=%d llm_calls=%d tool_rounds=%d tool_calls=%d exit=%s fallback=%s "
            "extraction=%s qualification=%s escalation=%s/%s grounding_violations=%d reply_source=%s handoff=%s",
            masked_sender,
            turn,
            diagnostics.llm_calls,
            diagnostics.tool_rounds,
            diagnostics.tool_calls_requested,
            diagnostics.loop_exit,
            diagnostics.fallback_used,
            diagnostics.extraction_source or "skipped",
            state.qualification.value,
            diagnostics.escalation_action,
            diagnostics.escalation_stage,
            diagnostics.grounding_violation_count,
            diagnostics.reply_source,
            diagnostics.handoff_outcome or "none",
        )

        # 16. Structured result.
        return AgentTurnResult(
            reply_text=reply_text,
            state_snapshot=state.model_copy(deep=True),
            tool_calls=list(ctx.records),
            diagnostics=diagnostics,
        )

    # -- Incoming guardrails ------------------------------------------------

    def _analyze_incoming(self, ctx: _TurnContext, masked_sender: str, turn: int) -> None:
        """Run the input-side detectors once on the customer text. Never raises.

        Each detector is contained separately: one that raises simply
        contributes no signal (``None``) and is recorded by exception type.
        The caller treats any detector failure as fail-closed for the turn.
        Detectors receive the text and bounded history only; they cannot
        mutate ``ConversationState``.
        """
        history = list(ctx.state.history)
        components: Dict[str, Any] = {}
        detectors = (
            ("injection", lambda: self._injection.detect(ctx.text)),
            ("anger", lambda: self._anger.score(ctx.text)),
            ("repetition", lambda: self._repetition.detect(ctx.text, history)),
            ("human_request", lambda: self._human_request.detect(ctx.text)),
        )
        for name, run in detectors:
            try:
                components[name] = run()
            except Exception as exc:  # detectors are pure; a crash is a bug, not the customer's problem
                error_type = type(exc).__name__
                ctx.guardrail_error_types.append(error_type)
                logger.error("Guardrail detector %s failed for %s (turn %d): %s", name, masked_sender, turn, error_type)
                logger.debug("Guardrail detector failure detail", exc_info=exc)
        try:
            ctx.signals = GuardrailSignals(**components)
        except Exception as exc:  # a detector returned the wrong shape
            error_type = type(exc).__name__
            ctx.guardrail_error_types.append(error_type)
            ctx.signals = GuardrailSignals()
            logger.error("Guardrail signals invalid for %s (turn %d): %s", masked_sender, turn, error_type)
            logger.debug("Guardrail signal failure detail", exc_info=exc)

    # -- Escalation policy --------------------------------------------------

    def _evaluate_policy(
        self, ctx: _TurnContext, signals: GuardrailSignals, stage: EscalationStage
    ) -> EscalationDecision:
        """Ask the pure policy for a verdict. Fails closed (escalate) on error."""
        try:
            decision = self._policy.evaluate(signals, ctx.state)
            if not isinstance(decision, EscalationDecision):
                raise TypeError("escalation policy returned a non-EscalationDecision")
        except Exception as exc:  # the policy is pure; failing open is not an option
            ctx.policy_error_type = type(exc).__name__
            logger.error("Escalation policy failed (%s): %s; failing closed", stage, ctx.policy_error_type)
            logger.debug("Escalation policy failure detail", exc_info=exc)
            decision = _POLICY_ERROR_DECISION
        ctx.decision = decision
        ctx.decision_stage = stage
        return decision

    def _apply_decision(self, ctx: _TurnContext, decision: EscalationDecision) -> None:
        """Apply the state transition a decision implies. Python only; never the model.

        Called once per turn, with the decision the turn acts on; that
        decision is also what ``_submit_handoff`` hands to the adapter later.
        """
        if decision.action == EscalationAction.ESCALATE and ctx.state.escalation.status == EscalationStatus.NONE:
            ctx.state.mark_escalated(_escalation_reason(decision))
        # ``handoff_ready`` is reported, not transitioned: ``mark_handoff_ready``
        # is an explicit consent step owned by a later slice, and a sink
        # accepting a request is not a human taking over (``mark_handed_off``).
        ctx.applied_decision = decision

    # -- Human handoff ------------------------------------------------------

    def _submit_handoff(self, ctx: _TurnContext, masked_sender: str, turn: int) -> None:
        """Hand the applied decision to the sink if it asks for a human. Never raises.

        The adapter (``handoff_request_from_decision``) decides whether the
        decision warrants a request at all and builds the whole payload;
        the sink owns deduplication. This method submits at most once and
        never retries: a failure is contained, recorded by exception type,
        and the turn's deterministic reply and state transition stand.
        """
        decision = ctx.applied_decision
        if decision is None:
            return  # a guardrail/LLM failure ended the turn before any decision was applied
        try:
            request = handoff_request_from_decision(decision, ctx.state)
        except Exception as exc:  # the adapter is pure; a crash is a bug, not the customer's problem
            ctx.handoff_outcome = "unavailable"
            ctx.handoff_error_type = type(exc).__name__
            logger.error("Handoff request build failed for %s (turn %d): %s", masked_sender, turn, ctx.handoff_error_type)
            logger.debug("Handoff request build failure detail", exc_info=exc)
            return
        if request is None:
            return  # the policy did not ask for a human

        ctx.handoff_kind = request.kind.value
        ctx.handoff_priority = request.priority.value
        if self._handoff_sink is None:
            ctx.handoff_outcome = "unavailable"
            logger.warning(
                "No handoff sink configured: %s handoff for %s (turn %d) is recorded in state only",
                ctx.handoff_kind,
                masked_sender,
                turn,
            )
            return

        ctx.handoff_attempted = True
        try:
            result = self._handoff_sink.submit(request)
            if not isinstance(result, HandoffResult):
                raise TypeError("handoff sink returned a non-HandoffResult")
        except Exception as exc:  # transport or programming failure: contain, never expose
            ctx.handoff_outcome = "unavailable"
            ctx.handoff_error_type = type(exc).__name__
            logger.error("Handoff submission failed for %s (turn %d): %s", masked_sender, turn, ctx.handoff_error_type)
            logger.debug("Handoff submission failure detail", exc_info=exc)
            return

        ctx.handoff_accepted = result.accepted
        ctx.handoff_outcome = result.outcome.value
        ctx.handoff_id = result.handoff_id
        logger.info(
            "Handoff %s for %s (turn %d): kind=%s priority=%s id=%s",
            ctx.handoff_outcome,
            masked_sender,
            turn,
            ctx.handoff_kind,
            ctx.handoff_priority,
            ctx.handoff_id,
        )

    # -- Lead extraction ----------------------------------------------------

    async def _update_lead(self, ctx: _TurnContext, masked_sender: str, turn: int) -> None:
        """Extract lead data from the current message and merge it into state.

        Exactly one extraction per customer turn, never per tool round. The
        extractor is pure: its delta reaches the profile only through
        ``ConversationState.apply_lead_delta``, which owns merge/provenance
        and recomputes qualification deterministically. Never raises — an
        extractor that fails, returns fallback, or crashes leaves the lead
        profile untouched and the turn continues normally.
        """
        if self._extractor is None:
            return
        ctx.extraction_attempted = True
        try:
            result: ExtractionResult = await self._extractor.extract(
                ctx.text,
                state=ctx.state,
                profile=ctx.state.lead,
            )
            ctx.extraction_source = result.source if result.source in _EXTRACTION_SOURCES else None
            ctx.extraction_errors_count = len(result.errors)
            if not result.success:
                logger.info("Lead extraction for %s (turn %d) fell back; lead state unchanged", masked_sender, turn)
                return
            # The ONLY point where extracted data enters state.
            qualification = ctx.state.apply_lead_delta(result.delta)
            ctx.extraction_success = True
            ctx.extracted_field_names = list(result.delta.provided_fields().keys())
            logger.info(
                "Lead extraction for %s (turn %d): fields=%s qualification=%s",
                masked_sender,
                turn,
                ctx.extracted_field_names,
                qualification.value,
            )
        except Exception as exc:  # extraction is enrichment; it must never fail the turn
            ctx.extraction_error_type = type(exc).__name__
            logger.error(
                "Lead extraction failed for %s (turn %d): %s; lead state unchanged",
                masked_sender,
                turn,
                ctx.extraction_error_type,
            )
            logger.debug("Lead extraction failure detail", exc_info=exc)

    # -- Reply generation + outgoing grounding ------------------------------

    async def _generate_reply(self, ctx: _TurnContext) -> "tuple[str, LoopExit]":
        """LLM/tool loop, then grounding and policy on every candidate reply.

        Returns the final customer-facing text. A candidate the validator
        rejects is never returned; at most ``MAX_CORRECTIVE_GENERATIONS``
        tool-free rewrites are tried, then the deterministic recovery reply.
        """
        candidate, loop_exit = await self._run_turn(ctx)
        if not candidate:
            ctx.reply_source = "fallback"
            ctx.fallback_reason = "empty_reply"
            return SAFE_FALLBACK_REPLY, "empty_reply"

        if self._accept_candidate(ctx, candidate):
            ctx.reply_source = "model"
            return candidate, loop_exit

        # A rewrite can only cure a grounding problem (``suppress``). Any other
        # blocking verdict on the outgoing stage (``escalate`` on a policy
        # error, for instance) does not depend on the candidate text, so a
        # rewrite could not change it: skip the model call and go straight
        # to the deterministic reply.
        rewrite_may_help = ctx.decision is not None and ctx.decision.action == EscalationAction.SUPPRESS
        for _ in range(MAX_CORRECTIVE_GENERATIONS if rewrite_may_help else 0):
            if ctx.grounding_error_type is not None:
                break  # a broken validator cannot approve a rewrite either
            ctx.corrective_generation_attempted = True
            ctx.transcript.append(ChatMessage(role="assistant", content=candidate))
            ctx.transcript.append(
                ChatMessage(
                    role="system",
                    content=CORRECTIVE_INSTRUCTION.format(codes=", ".join(ctx.grounding_reason_codes) or "unverified"),
                )
            )
            # Text-only, tools never offered: the rewrite may not go looking
            # for new facts, only drop the unsupported ones.
            response = await self._call_llm(ctx, tools=None)
            candidate = response.content or ""
            if candidate and self._accept_candidate(ctx, candidate):
                ctx.reply_source = "model_corrected"
                return candidate, loop_exit

        # Still unsafe (or unverifiable): the deterministic recovery action.
        ctx.reply_source = "fallback"
        ctx.fallback_reason = "ungrounded_reply"
        decision = ctx.decision if ctx.decision is not None else _POLICY_ERROR_DECISION
        self._apply_decision(ctx, decision)
        return _deterministic_reply(decision), loop_exit

    def _accept_candidate(self, ctx: _TurnContext, candidate: str) -> bool:
        """Grounding check + outgoing policy for one candidate reply.

        The policy sees a grounding-only signal bundle plus the current state
        (qualification, escalation, flags), so the incoming detectors' counts
        are not applied a second time. Only a rejected candidate bumps
        ``grounding_violations``.
        """
        grounding = self._validate_grounding(ctx, candidate)
        decision = self._evaluate_policy(ctx, GuardrailSignals(grounding=grounding), stage="outgoing")
        if decision.action in _BLOCKING_ACTIONS:
            ctx.grounding_violation_count += 1
            ctx.state.flags = apply_signals_to_flags(ctx.state.flags, GuardrailSignals(grounding=grounding))
            return False
        self._apply_decision(ctx, decision)
        return True

    def _validate_grounding(self, ctx: _TurnContext, candidate: str) -> GroundingResult:
        """Check a candidate against current-turn tool results and the knowledge base.

        A validator error is a verdict of *ungrounded*: an unvalidated model
        reply is never sent.
        """
        ctx.grounding_checks += 1
        try:
            result = self._grounding.validate(
                candidate,
                tool_results=ctx.state.current_turn_tool_results,
                knowledge=self._knowledge,
            )
            if not isinstance(result, GroundingResult):
                raise TypeError("grounding validator returned a non-GroundingResult")
        except Exception as exc:
            ctx.grounding_error_type = type(exc).__name__
            logger.error("Grounding validation failed: %s; treating reply as ungrounded", ctx.grounding_error_type)
            logger.debug("Grounding validation failure detail", exc_info=exc)
            result = _ungrounded_by_error()
        for code in result.reason_codes:
            if code not in ctx.grounding_reason_codes:
                ctx.grounding_reason_codes.append(code)
        return result

    # -- Turn pipeline ------------------------------------------------------

    async def _run_turn(self, ctx: _TurnContext) -> "tuple[Optional[str], LoopExit]":
        """Bounded LLM/tool loop. Returns the reply text (or ``None``) and why the loop ended."""
        tool_specs = self._llm_tool_specs()

        for round_index in range(1, self._max_tool_rounds + 1):
            if not tool_specs:
                break  # nothing to offer; go straight to the text-only call
            response = await self._call_llm(ctx, tools=tool_specs)
            if not response.has_tool_calls:
                return response.content, "text_reply"

            ctx.tool_rounds += 1
            self._handle_tool_round(ctx, response, round_index)

            if ctx.tools_closed is not None:
                break
            if round_index == self._max_tool_rounds:
                ctx.tool_round_limit_reached = True

        # Final call with tools omitted entirely (never tool_choice="none").
        ctx.final_call_tools_omitted = True
        response = await self._call_llm(ctx, tools=None)
        if ctx.tools_closed is not None:
            exit_reason: LoopExit = ctx.tools_closed
        elif ctx.tool_round_limit_reached:
            exit_reason = "tool_round_limit"
        else:
            exit_reason = "text_reply"
        return response.content, exit_reason

    async def _call_llm(self, ctx: _TurnContext, tools: Optional[List[LLMToolSpec]]) -> LLMResponse:
        messages = self._build_messages(ctx)
        ctx.llm_calls += 1
        if tools:
            response = await self._llm.complete(messages, tools=tools)
        else:
            # No ``tools`` kwarg at all: the provider then sends neither
            # ``tools`` nor ``tool_choice``.
            response = await self._llm.complete(messages)
        if response.usage is not None:
            ctx.usage_prompt_tokens += response.usage.prompt_tokens
            ctx.usage_completion_tokens += response.usage.completion_tokens
        return response

    def _build_messages(self, ctx: _TurnContext) -> List[ChatMessage]:
        """Prompt bundle (system, history, current message) + this turn's transcript."""
        tool_results = [
            invocation.model_dump(mode="json", exclude={"turn"})
            for invocation in ctx.state.current_turn_tool_results
        ]
        bundle = PromptBuilder.build(
            state=ctx.state,
            knowledge=self._knowledge,
            current_message=ctx.text,
            tool_results=tool_results or None,
            allowed_question=None,  # qualification policy is a later slice
        )
        return bundle.to_messages() + list(ctx.transcript)

    def _llm_tool_specs(self) -> List[LLMToolSpec]:
        """Registry schemas -> the ``ToolSpec`` shape ``complete()`` accepts."""
        specs: List[LLMToolSpec] = []
        for schema in self._tools.list_specs():
            function = schema["function"]
            specs.append(
                LLMToolSpec(
                    name=function["name"],
                    description=function["description"],
                    parameters=function["parameters"],
                )
            )
        return specs

    # -- Tool handling ------------------------------------------------------

    def _handle_tool_round(self, ctx: _TurnContext, response: LLMResponse, round_index: int) -> None:
        """Execute one response's tool calls sequentially and append the transcript."""
        call_ids = [
            tc.id or f"call_r{round_index}_{index}"
            for index, tc in enumerate(response.tool_calls, start=1)
        ]
        ctx.transcript.append(_assistant_tool_call_message(response, call_ids))

        for call_id, tool_call in zip(call_ids, response.tool_calls):
            record, result = self._execute_tool_call(ctx, tool_call, call_id, round_index)
            ctx.records.append(record)
            ctx.transcript.append(_tool_result_message(call_id, result))
            ctx.state.record_tool_invocation(
                ToolInvocation(
                    tool_name=record.tool_name,
                    turn=ctx.state.turn_count,
                    status=record.status,
                    ok=record.ok,
                    arguments=record.arguments,
                    result=result,
                )
            )

    def _execute_tool_call(
        self,
        ctx: _TurnContext,
        tool_call: ToolCall,
        call_id: str,
        round_index: int,
    ) -> "tuple[ToolCallRecord, Dict[str, Any]]":
        """Validate, budget-check and (if allowed) execute one tool call.

        Always returns a structured result to feed back to the model, even
        when the call is rejected, so every tool-call ID gets an answer.
        """
        tool_name = _safe_tool_name(tool_call.name)
        arguments = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}

        def rejected(disposition: ToolCallDisposition, status: str, code: str, message: str):
            result = _error_result(status, code, message)
            record = ToolCallRecord(
                call_id=call_id,
                tool_name=tool_name,
                round=round_index,
                disposition=disposition,
                status=status,
                ok=False,
                arguments=arguments,
                raw_arguments_preview=_preview(tool_call.raw_arguments),
                error_code=code,
            )
            return record, result

        if ctx.tools_closed == "tool_unavailable":
            return rejected(
                "rejected_tool_unavailable",
                STATUS_UNAVAILABLE,
                "tool_unavailable_this_turn",
                "A tool was unavailable earlier this turn; not retrying.",
            )
        if ctx.invalid_input_budget_exhausted:
            return rejected(
                "rejected_retry_budget",
                STATUS_INVALID_INPUT,
                "retry_limit_reached",
                "Tool arguments were invalid too many times this turn; answer without the tool.",
            )
        if self._tools.get(tool_call.name) is None:
            ctx.invalid_input_failures += 1
            self._close_tools_if_exhausted(ctx)
            return rejected(
                "rejected_unknown_tool",
                STATUS_INVALID_INPUT,
                "unknown_tool",
                "No such tool is available.",
            )
        if not tool_call.is_valid:
            ctx.invalid_input_failures += 1
            self._close_tools_if_exhausted(ctx)
            return rejected(
                "rejected_malformed_arguments",
                STATUS_INVALID_INPUT,
                "malformed_arguments_json",
                "Tool arguments were not a valid JSON object.",
            )

        try:
            result = self._tools.execute(tool_call.name, arguments)
            if not isinstance(result, dict):
                raise TypeError("tool handler returned a non-dict result")
        except Exception as exc:  # handlers promise not to raise; belt and braces
            logger.error("Tool %s raised %s; treating as unavailable", tool_name, type(exc).__name__)
            logger.debug("Tool failure detail", exc_info=exc)
            result = _error_result(
                STATUS_UNAVAILABLE,
                "tool_execution_failed",
                "The tool could not process the request right now.",
            )

        status = _result_status(result)
        if status == STATUS_INVALID_INPUT:
            ctx.invalid_input_failures += 1
            self._close_tools_if_exhausted(ctx)
        elif status == STATUS_UNAVAILABLE:
            ctx.tools_closed = "tool_unavailable"

        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            round=round_index,
            disposition="executed",
            status=status,
            ok=status in (STATUS_OK, STATUS_NO_MATCH),
            arguments=arguments,
            raw_arguments_preview=_preview(tool_call.raw_arguments),
            error_code=_result_error_code(result),
        )
        return record, result

    @staticmethod
    def _close_tools_if_exhausted(ctx: _TurnContext) -> None:
        if ctx.invalid_input_budget_exhausted and ctx.tools_closed is None:
            ctx.tools_closed = "invalid_input_retries_exhausted"

    # -- Diagnostics --------------------------------------------------------

    @staticmethod
    def _diagnostics(
        ctx: _TurnContext,
        masked_sender: str,
        message_id: Optional[str],
        turn: int,
        loop_exit: LoopExit,
        error_type: Optional[str],
    ) -> TurnDiagnostics:
        executed = sum(1 for r in ctx.records if r.disposition == "executed")
        signals = ctx.signals
        decision = ctx.decision
        return TurnDiagnostics(
            sender=masked_sender,
            message_id=message_id,
            turn=turn,
            llm_calls=ctx.llm_calls,
            tool_rounds=ctx.tool_rounds,
            tool_calls_requested=len(ctx.records),
            tool_calls_executed=executed,
            tool_calls_rejected=len(ctx.records) - executed,
            tool_round_limit_reached=ctx.tool_round_limit_reached,
            final_call_tools_omitted=ctx.final_call_tools_omitted,
            loop_exit=loop_exit,
            fallback_used=ctx.fallback_reason is not None,
            fallback_reason=ctx.fallback_reason,
            error_type=error_type,
            usage_prompt_tokens=ctx.usage_prompt_tokens,
            usage_completion_tokens=ctx.usage_completion_tokens,
            extraction_attempted=ctx.extraction_attempted,
            extraction_success=ctx.extraction_success,
            extraction_source=ctx.extraction_source,
            extracted_field_names=list(ctx.extracted_field_names),
            extraction_errors_count=ctx.extraction_errors_count,
            extraction_error_type=ctx.extraction_error_type,
            injection_suspected=bool(signals.injection is not None and signals.injection.suspected),
            injection_hit_count=signals.injection.hit_count if signals.injection is not None else 0,
            anger_score=signals.anger.score if signals.anger is not None else 0.0,
            repetition_detected=bool(signals.repetition is not None and signals.repetition.repeated),
            human_requested=bool(signals.human_request is not None and signals.human_request.requested),
            guardrail_error_types=list(ctx.guardrail_error_types),
            grounding_checks=ctx.grounding_checks,
            grounding_violation_count=ctx.grounding_violation_count,
            grounding_reason_codes=list(ctx.grounding_reason_codes),
            grounding_error_type=ctx.grounding_error_type,
            corrective_generation_attempted=ctx.corrective_generation_attempted,
            escalation_action=decision.action.value if decision is not None else None,
            escalation_reason_codes=list(decision.reason_codes) if decision is not None else [],
            escalation_stage=ctx.decision_stage,
            policy_error_type=ctx.policy_error_type,
            reply_source=ctx.reply_source,
            handoff_attempted=ctx.handoff_attempted,
            handoff_accepted=ctx.handoff_accepted,
            handoff_outcome=ctx.handoff_outcome,
            handoff_id=ctx.handoff_id,
            handoff_kind=ctx.handoff_kind,
            handoff_priority=ctx.handoff_priority,
            handoff_error_type=ctx.handoff_error_type,
        )


__all__ = [
    "AGENT_MAX_TOOL_ROUNDS",
    "AgentOrchestrator",
    "AgentTurnResult",
    "CLARIFICATION_REPLY",
    "CORRECTIVE_INSTRUCTION",
    "HUMAN_HANDOFF_REPLY",
    "MAX_CORRECTIVE_GENERATIONS",
    "MAX_INVALID_INPUT_RETRIES",
    "SAFE_FALLBACK_REPLY",
    "SAFE_REFUSAL_REPLY",
    "ToolCallRecord",
    "TurnDiagnostics",
    "UNVERIFIED_RECOVERY_REPLY",
]
