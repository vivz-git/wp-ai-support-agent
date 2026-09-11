"""Deterministic agent orchestrator for the AI WhatsApp Support Agent.

``AgentOrchestrator.handle_turn`` coordinates one customer turn across the
pieces built in earlier Milestone 2 slices:

    ConversationStore  -> load / save ``ConversationState``
    PromptBuilder      -> business-aware messages for the LLM
    LLMProvider        -> ``complete(messages, tools=...)`` (native tool calling)
    ToolRegistry       -> tool specs + deterministic tool execution

Architectural principle: the LLM is NOT the workflow controller. Python owns
state, validation, tool execution, tool-call limits, error handling,
persistence and the turn result. The LLM only supplies natural language and
tool-call requests.

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
- No guardrails, escalation, lead extraction or qualification policy here;
  those are later slices.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.agent.prompts import PromptBuilder
from app.agent.state import ConversationState, MAX_MESSAGE_LENGTH, ToolInvocation
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

SAFE_FALLBACK_REPLY = "Sorry — I’m having trouble checking that right now. Let me get the team to help."

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
]

FallbackReason = Literal["empty_reply", "llm_error"]


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

    Never contains prompt text, credentials, authorization headers, or an
    unmasked phone number.
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
    # Tool-protocol messages appended after the current customer message:
    # assistant(tool_calls) followed by one role="tool" message per call.
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


# ---------------------------------------------------------------------------
# AgentOrchestrator
# ---------------------------------------------------------------------------


class AgentOrchestrator:
    """Coordinates one conversation turn: state -> prompt -> LLM -> tools -> reply.

    The orchestrator is a coordinator, not a store: all history lives in
    ``ConversationState`` (persisted through ``ConversationStore``), and all
    prompt text comes from ``PromptBuilder``.
    """

    def __init__(
        self,
        llm: LLMProvider,
        knowledge: KnowledgeBase,
        tools: ToolRegistry,
        store: ConversationStore,
        max_tool_rounds: int = AGENT_MAX_TOOL_ROUNDS,
    ):
        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds must not be negative")
        self._llm = llm
        self._knowledge = knowledge
        self._tools = tools
        self._store = store
        self._max_tool_rounds = max_tool_rounds

    # -- Public API ---------------------------------------------------------

    async def handle_turn(
        self,
        sender_id: str,
        text: str,
        message_id: Optional[str] = None,
    ) -> AgentTurnResult:
        """Run one customer turn end to end and return a structured result.

        Never raises for LLM or tool failures. Raises ``ValueError`` only
        for caller misuse (blank ``text``), before any state is touched.
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
        # LLM loop is done: PromptBuilder renders the current message itself,
        # delimited, as the final user message, so adding it to ``history``
        # first would send it twice.
        turn = state.begin_turn()

        ctx = _TurnContext(state=state, text=customer_text)
        masked_sender = mask_phone_number(sender_id)

        # 3-9. Prompt -> LLM -> tools -> reply, with a deterministic fallback.
        fallback_reason: Optional[FallbackReason] = None
        error_type: Optional[str] = None
        try:
            reply_text, loop_exit = await self._run_turn(ctx)
            if not reply_text:
                reply_text = SAFE_FALLBACK_REPLY
                fallback_reason = "empty_reply"
                loop_exit = "empty_reply"
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
            fallback_reason = "llm_error"
            loop_exit = "llm_error"

        # 10. Persist: this turn's messages, tool history and any other state changes.
        state.add_user_message(customer_text)
        state.add_assistant_message(reply_text)
        self._store.save(state)

        executed = sum(1 for r in ctx.records if r.disposition == "executed")
        diagnostics = TurnDiagnostics(
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
            fallback_used=fallback_reason is not None,
            fallback_reason=fallback_reason,
            error_type=error_type,
            usage_prompt_tokens=ctx.usage_prompt_tokens,
            usage_completion_tokens=ctx.usage_completion_tokens,
        )
        logger.info(
            "Agent turn complete for %s: turn=%d llm_calls=%d tool_rounds=%d tool_calls=%d exit=%s fallback=%s",
            masked_sender,
            turn,
            diagnostics.llm_calls,
            diagnostics.tool_rounds,
            diagnostics.tool_calls_requested,
            diagnostics.loop_exit,
            diagnostics.fallback_used,
        )

        # 11. Structured result.
        return AgentTurnResult(
            reply_text=reply_text,
            state_snapshot=state.model_copy(deep=True),
            tool_calls=list(ctx.records),
            diagnostics=diagnostics,
        )

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
        """Prompt bundle (system, history, current message) + this turn's tool transcript."""
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


__all__ = [
    "AGENT_MAX_TOOL_ROUNDS",
    "AgentOrchestrator",
    "AgentTurnResult",
    "MAX_INVALID_INPUT_RETRIES",
    "SAFE_FALLBACK_REPLY",
    "ToolCallRecord",
    "TurnDiagnostics",
]
