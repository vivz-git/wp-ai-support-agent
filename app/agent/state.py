"""Conversation state model for the AI WhatsApp Support Agent.

``ConversationState`` is the single, validated record of one sender's
conversation: what they said, what we said, what we think they want, what
we know about them as a lead, which tools ran, and the behavioural
signals later guardrail/escalation slices will act on.

Design constraints (Milestone 2, Slice 4):
- Pure data model. No orchestrator, no prompt, no LLM, no network.
- Not wired into ``app.main`` — the Milestone 1 request path is untouched.
- The state evolves across turns, so it is mutable, but every mutation
  path re-validates (``validate_assignment=True``) and the bounded
  collections are trimmed on both append and load.
- Qualification is never assigned directly from outside this module; it is
  recomputed from the ``LeadProfile`` via ``evaluate_qualification`` or
  moved by the explicit ``mark_*`` transitions.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.agent.lead import (
    LeadDelta,
    LeadProfile,
    LeadSource,
    QualificationState,
    STICKY_QUALIFICATION_STATES,
    evaluate_qualification,
    merge_lead_delta,
)
from app.llm.base import ChatMessage

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

MAX_INTENT_HISTORY = 20
MAX_CHAT_HISTORY = 20
MAX_TOOL_HISTORY = 20
MAX_KNOWN_FACTS = 50
MAX_MESSAGE_LENGTH = 4096  # WhatsApp text body limit
STORE_CAPACITY = 500


class InvalidTransitionError(ValueError):
    """Raised when a qualification/escalation transition is not allowed."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    FAQ = "faq"
    PRODUCT_INQUIRY = "product_inquiry"
    RECOMMENDATION = "recommendation"
    PRICE_CHECK = "price_check"
    AVAILABILITY_CHECK = "availability_check"
    WHOLESALE_INQUIRY = "wholesale_inquiry"
    ORDER_STATUS = "order_status"
    COMPLAINT = "complaint"
    HUMAN_REQUEST = "human_request"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"
    UNCLEAR = "unclear"


class EscalationStatus(str, Enum):
    NONE = "none"
    PENDING = "pending"
    HANDED_OFF = "handed_off"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class IntentRecord(BaseModel):
    """One classified intent, kept in the bounded intent history."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    turn: int = Field(..., ge=0)


class HistoryMessage(BaseModel):
    """One chat turn as stored in ``ConversationState.history``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    turn: int = Field(..., ge=0)

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.content)


class ToolInvocation(BaseModel):
    """A record of one tool call made on behalf of this conversation.

    ``result`` is kept only for current-turn results (the orchestrator will
    need it to ground a reply); history entries are stored with the result
    dropped so the bounded history stays small and serializable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_name: str = Field(..., min_length=1, max_length=64)
    turn: int = Field(..., ge=0)
    status: str = Field(..., min_length=1, max_length=32)
    ok: bool
    arguments: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None

    def without_result(self) -> "ToolInvocation":
        return self.model_copy(update={"result": None})


class EscalationState(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    status: EscalationStatus = EscalationStatus.NONE
    reason: Optional[str] = Field(None, max_length=200)
    requested_at_turn: Optional[int] = Field(None, ge=0)
    handed_off_at_turn: Optional[int] = Field(None, ge=0)


class ConversationFlags(BaseModel):
    """Deterministic behavioural counters for later guardrail/escalation rules.

    These are plain counters and scores. Nothing here decides anything;
    the rules that read them are a later slice.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    injection_suspected: bool = False
    injection_hits: int = Field(0, ge=0)
    anger_score: float = Field(0.0, ge=0.0, le=1.0)
    unanswered_asks: int = Field(0, ge=0)
    declines: int = Field(0, ge=0)
    repeated_question_count: int = Field(0, ge=0)
    off_topic_count: int = Field(0, ge=0)
    tool_failures_this_turn: int = Field(0, ge=0)
    grounding_violations: int = Field(0, ge=0)


# ---------------------------------------------------------------------------
# ConversationState
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _looks_like_whatsapp_number(sender_id: str) -> bool:
    return sender_id.isdigit() and 6 <= len(sender_id) <= 20


class ConversationState(BaseModel):
    """The full, validated state of one sender's conversation.

    Construct with ``ConversationState.new(sender_id)`` for a safe default,
    or ``model_validate`` on a previously ``model_dump(mode="json")``-ed
    dict to restore it.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    sender_id: str = Field(..., min_length=1, max_length=64)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    turn_count: int = Field(0, ge=0)
    language: str = Field("en", min_length=2, max_length=10)

    # Intent
    current_intent: Intent = Intent.UNCLEAR
    intent_confidence: float = Field(0.0, ge=0.0, le=1.0)
    intent_history: List[IntentRecord] = Field(default_factory=list)

    # Lead + qualification
    lead: LeadProfile = Field(default_factory=LeadProfile)
    qualification: QualificationState = QualificationState.UNKNOWN

    # Knowledge
    known_facts: Dict[str, str] = Field(default_factory=dict)

    # History
    history: List[HistoryMessage] = Field(default_factory=list)

    # Tools
    tool_history: List[ToolInvocation] = Field(default_factory=list)
    current_turn_tool_results: List[ToolInvocation] = Field(default_factory=list)

    # Escalation + flags
    escalation: EscalationState = Field(default_factory=EscalationState)
    flags: ConversationFlags = Field(default_factory=ConversationFlags)

    # -- Validation ---------------------------------------------------------

    @field_validator("sender_id", "language")
    @classmethod
    def _strip_required_text(cls, value: str, info) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be blank")
        return stripped

    @field_validator("intent_history")
    @classmethod
    def _bound_intent_history(cls, value: List[IntentRecord]) -> List[IntentRecord]:
        return value[-MAX_INTENT_HISTORY:]

    @field_validator("history")
    @classmethod
    def _bound_history(cls, value: List[HistoryMessage]) -> List[HistoryMessage]:
        return value[-MAX_CHAT_HISTORY:]

    @field_validator("tool_history")
    @classmethod
    def _bound_tool_history(cls, value: List[ToolInvocation]) -> List[ToolInvocation]:
        return value[-MAX_TOOL_HISTORY:]

    @field_validator("known_facts")
    @classmethod
    def _validate_known_facts(cls, value: Dict[str, str]) -> Dict[str, str]:
        if len(value) > MAX_KNOWN_FACTS:
            raise ValueError(f"known_facts may hold at most {MAX_KNOWN_FACTS} entries")
        for key, fact in value.items():
            if not key.strip() or len(key) > 64:
                raise ValueError("known_facts keys must be non-blank and at most 64 characters")
            if len(fact) > 500:
                raise ValueError(f"known_facts['{key}'] must be at most 500 characters")
        return value

    @model_validator(mode="after")
    def _timestamps_ordered(self) -> "ConversationState":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")
        return self

    # -- Construction -------------------------------------------------------

    @classmethod
    def new(cls, sender_id: str, language: str = "en") -> "ConversationState":
        """Safe default state for a sender.

        The WhatsApp number is webhook metadata: when ``sender_id`` is a
        plain wa_id it is copied into the lead profile so no later step ever
        needs to ask the customer for it.
        """
        lead = LeadProfile(source=LeadSource.WHATSAPP)
        if _looks_like_whatsapp_number(sender_id):
            lead.whatsapp_number = sender_id
        return cls(sender_id=sender_id, language=language, lead=lead)

    # -- Turn lifecycle -----------------------------------------------------

    def begin_turn(self) -> int:
        """Start a new turn: bump the counter and clear per-turn scratch."""
        self.turn_count = self.turn_count + 1
        self.current_turn_tool_results = []
        self.flags.tool_failures_this_turn = 0
        self._touch()
        return self.turn_count

    def _touch(self) -> None:
        self.updated_at = _utcnow()

    # -- Intent -------------------------------------------------------------

    def set_intent(self, intent: Intent, confidence: float) -> IntentRecord:
        record = IntentRecord(intent=intent, confidence=confidence, turn=self.turn_count)
        self.current_intent = record.intent
        self.intent_confidence = record.confidence
        self.intent_history = self.intent_history + [record]  # re-validated → bounded
        self._touch()
        return record

    # -- History ------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._append_history("user", content)

    def add_assistant_message(self, content: str) -> None:
        self._append_history("assistant", content)

    def _append_history(self, role: Literal["user", "assistant"], content: str) -> None:
        message = HistoryMessage(role=role, content=content, turn=self.turn_count)
        self.history = self.history + [message]  # re-validated → bounded
        self._touch()

    def chat_messages(self) -> List[ChatMessage]:
        """The bounded history as ``ChatMessage``s, ready for an LLM call."""
        return [m.to_chat_message() for m in self.history]

    # -- Tools --------------------------------------------------------------

    def record_tool_invocation(self, invocation: ToolInvocation) -> None:
        """Record a tool call for this turn and in the bounded history."""
        self.current_turn_tool_results = self.current_turn_tool_results + [invocation]
        self.tool_history = self.tool_history + [invocation.without_result()]
        if not invocation.ok:
            self.flags.tool_failures_this_turn = self.flags.tool_failures_this_turn + 1
        self._touch()

    # -- Knowledge ----------------------------------------------------------

    def remember_fact(self, key: str, value: str) -> None:
        """Store a known fact; the newest fact wins when the cap is reached."""
        facts = dict(self.known_facts)
        facts.pop(key, None)
        if len(facts) >= MAX_KNOWN_FACTS:
            oldest_key = next(iter(facts))
            del facts[oldest_key]
        facts[key] = value
        self.known_facts = facts
        self._touch()

    # -- Lead + qualification -----------------------------------------------

    def apply_lead_delta(self, delta: LeadDelta) -> QualificationState:
        """Merge a delta into the lead profile and re-evaluate qualification.

        This is the only path by which extracted data reaches the profile,
        and qualification is recomputed here rather than accepted from the
        delta — ``LeadDelta`` has no qualification field at all.
        """
        self.lead = merge_lead_delta(self.lead, delta, turn=self.turn_count)
        return self.reevaluate_qualification()

    def reevaluate_qualification(self) -> QualificationState:
        """Recompute the data-driven qualification state from the profile."""
        self.qualification = evaluate_qualification(self.lead, self.qualification, self.turn_count)
        self._touch()
        return self.qualification

    def mark_handoff_ready(self) -> QualificationState:
        """Explicit ``qualified -> handoff_ready`` transition.

        Requires a qualified profile; this cannot be used to skip
        qualification.
        """
        if self.qualification != QualificationState.QUALIFIED:
            raise InvalidTransitionError(
                f"handoff_ready requires state 'qualified', current is '{self.qualification.value}'"
            )
        if not self.lead.is_complete():
            raise InvalidTransitionError("handoff_ready requires a complete lead profile")
        self.qualification = QualificationState.HANDOFF_READY
        self._touch()
        return self.qualification

    def mark_declined(self) -> QualificationState:
        """The customer declined to share details or to be contacted."""
        if self.qualification == QualificationState.ESCALATED:
            raise InvalidTransitionError("cannot decline an escalated conversation")
        self.qualification = QualificationState.DECLINED
        self.flags.declines = self.flags.declines + 1
        self._touch()
        return self.qualification

    def mark_escalated(self, reason: str) -> QualificationState:
        """Route the conversation to a human; allowed from any state."""
        self.qualification = QualificationState.ESCALATED
        self.escalation = EscalationState(
            status=EscalationStatus.PENDING,
            reason=reason,
            requested_at_turn=self.turn_count,
            handed_off_at_turn=self.escalation.handed_off_at_turn,
        )
        self._touch()
        return self.qualification

    def mark_handed_off(self) -> EscalationStatus:
        """Record that a human has taken over (from a pending escalation or
        a handoff-ready lead)."""
        allowed = self.escalation.status == EscalationStatus.PENDING or (
            self.qualification == QualificationState.HANDOFF_READY
        )
        if not allowed:
            raise InvalidTransitionError(
                "handed_off requires a pending escalation or a handoff_ready lead"
            )
        self.escalation = EscalationState(
            status=EscalationStatus.HANDED_OFF,
            reason=self.escalation.reason,
            requested_at_turn=self.escalation.requested_at_turn,
            handed_off_at_turn=self.turn_count,
        )
        self._touch()
        return self.escalation.status

    @property
    def is_terminal(self) -> bool:
        """Whether the lead flow has reached a sticky end state."""
        return self.qualification in STICKY_QUALIFICATION_STATES


__all__ = [
    "ConversationFlags",
    "ConversationState",
    "EscalationState",
    "EscalationStatus",
    "HistoryMessage",
    "Intent",
    "IntentRecord",
    "InvalidTransitionError",
    "MAX_CHAT_HISTORY",
    "MAX_INTENT_HISTORY",
    "MAX_KNOWN_FACTS",
    "MAX_TOOL_HISTORY",
    "STORE_CAPACITY",
    "ToolInvocation",
]
