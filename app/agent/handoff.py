"""Human handoff boundary for the AI WhatsApp Support Agent.

This module is the seam between the agent and whatever human support system
eventually receives escalations. The agent side produces a
``HandoffRequest``; a ``HandoffSink`` consumes it and answers with a
``HandoffResult``. The agent never learns whether the sink is Slack, email,
a CRM, a ticketing tool, a dashboard, or (as in this slice) a bounded
in-memory list.

Responsibilities, in order:

- ``handoff_request_from_decision(decision, state)``: adapter from the
  existing ``EscalationDecision`` (``app.agent.escalation``) to a
  ``HandoffRequest``. It *consumes* a decision; it never decides whether an
  escalation should happen. Only ``escalate`` and ``handoff_ready`` produce
  a request. It reads ``state`` and never mutates it.
- ``HandoffRequest``: the minimum a human needs to pick the conversation
  up — why, how urgent, what we know about the lead, and the bounded
  customer/assistant transcript. Nothing else. No prompts, no provider
  output, no tool internals, no credentials, no raw phone number.
- ``HandoffSink`` / ``InMemoryHandoffSink``: the consumer protocol and its
  only implementation for this slice. Deterministic, bounded, dedupes
  repeated submissions for the same active escalation, and reports
  problems as a rejected ``HandoffResult`` rather than by crashing the
  agent.

Design constraints (Milestone 2, Slice 11):
- No network, no external services, no settings/env access.
- Not wired into ``app.main`` or the orchestrator yet.
- No module-level sink instance: callers construct and own their sink.
"""

import hashlib
import logging
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.agent.escalation import EscalationAction, EscalationDecision
from app.agent.lead import LeadProfile, QualificationState
from app.agent.state import MAX_CHAT_HISTORY, ConversationState
from app.config import mask_phone_number

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

# The transcript can never exceed the state's own history bound; it is a
# view of that history, not a second copy of the conversation.
MAX_TRANSCRIPT_ENTRIES = MAX_CHAT_HISTORY
# A human skims a handoff; a single 4096-character WhatsApp message is
# truncated rather than reproduced in full.
MAX_TRANSCRIPT_MESSAGE_LENGTH = 1000
MAX_SUMMARY_LENGTH = 500
MAX_REASON_CODES = 16
MAX_REASON_CODE_LENGTH = 64
DEFAULT_MAX_HANDOFFS = 500

_TRUNCATION_MARKER = "…"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class HandoffPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class HandoffKind(str, Enum):
    """Why a human is being asked in: a problem, or a sales opportunity."""

    ESCALATION = "escalation"  # from EscalationAction.ESCALATE
    QUALIFIED_LEAD = "qualified_lead"  # from EscalationAction.HANDOFF_READY


class HandoffStatus(str, Enum):
    """Lifecycle of one stored handoff. Deliberately tiny."""

    PENDING = "pending"  # created, nobody has picked it up
    ACCEPTED = "accepted"  # a human has taken it
    CLOSED = "closed"  # done; eligible for eviction


class HandoffOutcome(str, Enum):
    """What ``submit`` did with a request."""

    CREATED = "created"
    DEDUPLICATED = "deduplicated"
    REJECTED = "rejected"


# Reason codes the escalation policy emits, mapped to human urgency. The
# highest priority across all codes on a decision wins. Codes not listed
# fall back per kind (``_DEFAULT_PRIORITY``) so a new policy rule can never
# produce an unprioritized handoff.
_REASON_PRIORITY: Dict[str, HandoffPriority] = {
    "high_anger_complaint": HandoffPriority.URGENT,
    "human_requested": HandoffPriority.HIGH,
    "repeated_unresolved": HandoffPriority.HIGH,
    "injection_repeated": HandoffPriority.MEDIUM,
    "already_escalated": HandoffPriority.MEDIUM,
    # Consent given (state is handoff_ready): the customer is waiting.
    "handoff_ready": HandoffPriority.MEDIUM,
    # Profile complete but consent not yet confirmed: worth a look, not urgent.
    "lead_qualified": HandoffPriority.LOW,
}
_DEFAULT_PRIORITY: Dict[HandoffKind, HandoffPriority] = {
    HandoffKind.ESCALATION: HandoffPriority.MEDIUM,
    HandoffKind.QUALIFIED_LEAD: HandoffPriority.LOW,
}
_PRIORITY_RANK: Dict[HandoffPriority, int] = {
    HandoffPriority.LOW: 0,
    HandoffPriority.MEDIUM: 1,
    HandoffPriority.HIGH: 2,
    HandoffPriority.URGENT: 3,
}
_ACTION_KIND: Dict[EscalationAction, HandoffKind] = {
    EscalationAction.ESCALATE: HandoffKind.ESCALATION,
    EscalationAction.HANDOFF_READY: HandoffKind.QUALIFIED_LEAD,
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class HandoffError(Exception):
    """Base class for handoff-layer errors. Carries a stable ``code``."""

    code = "handoff_error"

    def __init__(self, message: str, code: Optional[str] = None):
        super().__init__(message)
        if code is not None:
            self.code = code


class HandoffSinkError(HandoffError):
    """A sink could not process a request. Never carries request content."""

    code = "sink_error"


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def conversation_id_for(sender_id: str) -> str:
    """Stable correlation key for a sender that is not the raw number.

    A plain SHA-256 prefix: deterministic across processes, no secret
    involved, and enough to tell two senders apart. The human-readable form
    is ``mask_phone_number``; this one exists so the handoff can be matched
    back to a conversation without carrying the number itself.
    """
    digest = hashlib.sha256(sender_id.strip().encode("utf-8")).hexdigest()
    return f"conv_{digest[:16]}"


class LeadSnapshot(BaseModel):
    """What a human needs to know about the lead. Nothing else.

    Built only from validated ``LeadProfile`` fields. ``whatsapp_number``,
    ``field_provenance`` and ``source`` are deliberately absent: the number
    is PII the human reaches through the conversation, and the rest is
    agent bookkeeping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    lead_track: str
    qualification: str
    contact_name: Optional[str] = None
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    city: Optional[str] = None
    email: Optional[str] = None
    monthly_volume_kg: Optional[float] = None
    timeline: Optional[str] = None
    intent_summary: Optional[str] = None
    missing_required_fields: List[str] = Field(default_factory=list)

    @classmethod
    def from_lead(cls, lead: LeadProfile, qualification: QualificationState, redact: Callable[[str], str]) -> "LeadSnapshot":
        return cls(
            lead_track=lead.effective_track().value,
            qualification=qualification.value,
            contact_name=_redact_optional(lead.contact_name, redact),
            business_name=_redact_optional(lead.business_name, redact),
            business_type=lead.business_type.value if lead.business_type is not None else None,
            city=_redact_optional(lead.city, redact),
            email=lead.email,
            monthly_volume_kg=lead.monthly_volume_kg,
            timeline=lead.timeline.value if lead.timeline is not None else None,
            intent_summary=_redact_optional(lead.intent_summary, redact),
            missing_required_fields=list(lead.missing_required_fields()),
        )

    def is_empty(self) -> bool:
        """True when nothing beyond track/qualification is known."""
        return all(
            getattr(self, name) is None
            for name in (
                "contact_name",
                "business_name",
                "business_type",
                "city",
                "email",
                "monthly_volume_kg",
                "timeline",
                "intent_summary",
            )
        )


class TranscriptEntry(BaseModel):
    """One customer or assistant message, bounded and sanitized."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["customer", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_TRANSCRIPT_MESSAGE_LENGTH)
    turn: int = Field(..., ge=0)


class HandoffRequest(BaseModel):
    """Everything a human needs to take over a conversation, and nothing more.

    Immutable once built. ``conversation_id`` is a hash of the sender ID and
    ``sender_masked`` is the masked number; the raw number appears nowhere.
    ``policy_priority`` preserves the escalation policy's own 1..12 rank so
    the human system can see exactly which rule won; ``priority`` is the
    human-facing urgency derived from the reason codes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    conversation_id: str = Field(..., min_length=1, max_length=64)
    sender_masked: str = Field(..., min_length=1, max_length=32)
    kind: HandoffKind
    reason_codes: List[str] = Field(..., min_length=1, max_length=MAX_REASON_CODES)
    priority: HandoffPriority
    policy_priority: int = Field(..., ge=1)
    lead: LeadSnapshot
    transcript: List[TranscriptEntry] = Field(default_factory=list, max_length=MAX_TRANSCRIPT_ENTRIES)
    summary: str = Field(..., min_length=1, max_length=MAX_SUMMARY_LENGTH)
    turn: int = Field(..., ge=0)
    language: str = Field("en", min_length=2, max_length=10)
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("reason_codes")
    @classmethod
    def _reason_codes_are_short_codes(cls, value: List[str]) -> List[str]:
        for code in value:
            if not code.strip() or len(code) > MAX_REASON_CODE_LENGTH:
                raise ValueError("reason codes must be non-blank and at most 64 characters")
        return value

    @property
    def dedupe_key(self) -> str:
        """One open handoff per conversation per kind.

        The escalation policy re-evaluates every turn and keeps returning
        ``escalate`` (via ``already_escalated``) for the life of an
        escalation, so keying on the reason would open a fresh ticket the
        moment the code list changed. A conversation has one active
        escalation; it gets one ticket until that ticket is closed.
        """
        return f"{self.conversation_id}:{self.kind.value}"

    @property
    def primary_reason(self) -> str:
        return self.reason_codes[0]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class HandoffResult(BaseModel):
    """What the sink did. Provider-neutral by construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    accepted: bool
    outcome: HandoffOutcome
    handoff_id: Optional[str] = None
    status: Optional[HandoffStatus] = None
    reason: str = Field(..., min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def deduplicated(self) -> bool:
        return self.outcome == HandoffOutcome.DEDUPLICATED


class HandoffRecord(BaseModel):
    """A stored handoff: the immutable request plus its mutable lifecycle."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    handoff_id: str
    request: HandoffRequest
    status: HandoffStatus = HandoffStatus.PENDING
    created_at: datetime
    updated_at: datetime
    # How many times the same active escalation was submitted; a human sees
    # "asked 4 times" without four tickets existing.
    submission_count: int = Field(1, ge=1)


# ---------------------------------------------------------------------------
# Sanitization helpers
# ---------------------------------------------------------------------------


def _make_redactor(sender_id: str) -> Callable[[str], str]:
    """Replace any verbatim occurrence of the raw sender number with its mask.

    Customers often type their own number into the chat; it must not reach
    the human-facing payload in the clear.
    """
    raw = sender_id.strip()
    masked = mask_phone_number(raw)

    def redact(text: str) -> str:
        return text.replace(raw, masked) if raw and raw in text else text

    return redact


def _redact_optional(value: Optional[str], redact: Callable[[str], str]) -> Optional[str]:
    return redact(value) if value is not None else None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARKER)] + _TRUNCATION_MARKER


def _build_transcript(state: ConversationState, redact: Callable[[str], str]) -> List[TranscriptEntry]:
    """The bounded history as customer/assistant lines. No system, no tools."""
    entries: List[TranscriptEntry] = []
    for message in state.history[-MAX_TRANSCRIPT_ENTRIES:]:
        role: Literal["customer", "assistant"] = "customer" if message.role == "user" else "assistant"
        content = _truncate(redact(message.content), MAX_TRANSCRIPT_MESSAGE_LENGTH).strip()
        if not content:
            continue
        entries.append(TranscriptEntry(role=role, content=content, turn=message.turn))
    return entries


def _derive_priority(kind: HandoffKind, reason_codes: List[str]) -> HandoffPriority:
    best = _DEFAULT_PRIORITY[kind]
    for code in reason_codes:
        candidate = _REASON_PRIORITY.get(code)
        if candidate is not None and _PRIORITY_RANK[candidate] > _PRIORITY_RANK[best]:
            best = candidate
    return best


def _build_summary(kind: HandoffKind, reason_codes: List[str], lead: LeadSnapshot, sender_masked: str, turn: int) -> str:
    """One deterministic line a human can read before opening the transcript."""
    head = "Escalation" if kind == HandoffKind.ESCALATION else "Qualified lead"
    parts = [f"{head} for {sender_masked} after {turn} turn{'s' if turn != 1 else ''} ({', '.join(reason_codes)})."]
    who: List[str] = []
    if lead.contact_name:
        who.append(lead.contact_name)
    if lead.business_name:
        who.append(lead.business_name)
    if lead.business_type:
        who.append(lead.business_type)
    if lead.city:
        who.append(lead.city)
    if who:
        parts.append(f"{lead.lead_track.capitalize()} lead: {', '.join(who)}.")
    else:
        parts.append(f"Lead track: {lead.lead_track}; qualification: {lead.qualification}.")
    if lead.intent_summary:
        parts.append(f"Intent: {lead.intent_summary}")
    return _truncate(" ".join(parts), MAX_SUMMARY_LENGTH)


# ---------------------------------------------------------------------------
# Adapter: EscalationDecision -> HandoffRequest
# ---------------------------------------------------------------------------


def handoff_request_from_decision(
    decision: EscalationDecision,
    state: ConversationState,
    now: Optional[datetime] = None,
) -> Optional[HandoffRequest]:
    """Turn an escalation decision into a safe handoff request, or ``None``.

    Only ``escalate`` and ``handoff_ready`` produce a request; every other
    action (continue, clarify, refuse, suppress) returns ``None`` because
    the policy did not ask for a human. The decision's ``reason_codes`` and
    ``priority`` are carried through unchanged; ``state`` is read, never
    mutated, and nothing here consults settings, the environment, prompts,
    or provider output.
    """
    kind = _ACTION_KIND.get(decision.action)
    if kind is None:
        return None

    reason_codes = list(decision.reason_codes) or [decision.action.value]
    redact = _make_redactor(state.sender_id)
    sender_masked = mask_phone_number(state.sender_id)
    lead = LeadSnapshot.from_lead(state.lead, state.qualification, redact)
    transcript = _build_transcript(state, redact)

    return HandoffRequest(
        conversation_id=conversation_id_for(state.sender_id),
        sender_masked=sender_masked,
        kind=kind,
        reason_codes=reason_codes,
        priority=_derive_priority(kind, reason_codes),
        policy_priority=decision.priority,
        lead=lead,
        transcript=transcript,
        summary=_build_summary(kind, reason_codes, lead, sender_masked, state.turn_count),
        turn=state.turn_count,
        language=state.language,
        created_at=now if now is not None else _utcnow(),
    )


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class HandoffSink(Protocol):
    """Anything that can receive a ``HandoffRequest``.

    Implementations must return a ``HandoffResult`` for a well-formed
    request and must not raise for malformed input (they reject it). They
    may raise ``HandoffSinkError`` for transport-level failures; the caller
    treats that as "handoff unavailable" and replies safely. Programming
    errors are not caught here.
    """

    def submit(self, request: HandoffRequest) -> HandoffResult:  # pragma: no cover - protocol
        ...


# ---------------------------------------------------------------------------
# In-memory sink
# ---------------------------------------------------------------------------


class InMemoryHandoffSink:
    """Bounded, deterministic, in-process ``HandoffSink``.

    - Stores ``HandoffRecord``s in insertion order, capped at
      ``max_handoffs``; when full, the oldest *closed* record is evicted
      first, then the oldest of any status (with a warning), so memory is
      always bounded.
    - Dedupes on ``HandoffRequest.dedupe_key`` while a matching record is
      not closed: the existing handoff is returned and its
      ``submission_count`` bumped. Once closed, the next submission opens a
      new one.
    - IDs are ``ho-000001, ho-000002, ...`` from a per-sink counter, so a
      fresh sink fed the same requests yields the same IDs.
    - Returns copies from ``get``/``list``; stored records are never handed
      out by reference.
    - A ``threading.Lock`` serializes mutations. No async, no external
      dependency.
    """

    def __init__(self, max_handoffs: int = DEFAULT_MAX_HANDOFFS, clock: Optional[Callable[[], datetime]] = None):
        if max_handoffs < 1:
            raise ValueError("max_handoffs must be at least 1")
        self.max_handoffs = max_handoffs
        self._clock = clock or _utcnow
        self._records: "OrderedDict[str, HandoffRecord]" = OrderedDict()
        self._open_by_key: Dict[str, str] = {}  # dedupe_key -> handoff_id (non-closed only)
        self._sequence = 0
        self._lock = threading.Lock()

    # -- HandoffSink --------------------------------------------------------

    def submit(self, request: HandoffRequest) -> HandoffResult:
        """Create or reuse a handoff for ``request``. Never raises for bad input."""
        if not isinstance(request, HandoffRequest):
            logger.warning("Handoff sink rejected malformed request of type %s", type(request).__name__)
            return self._rejected("malformed_request")
        try:
            # Re-validate: a subclass or a monkeypatched instance cannot smuggle
            # an invalid payload past the boundary.
            HandoffRequest.model_validate(request.model_dump())
        except ValidationError:
            logger.warning("Handoff sink rejected request that failed validation")
            return self._rejected("invalid_request")

        with self._lock:
            now = self._clock()
            existing_id = self._open_by_key.get(request.dedupe_key)
            if existing_id is not None:
                record = self._records[existing_id]
                record.submission_count = record.submission_count + 1
                record.updated_at = now
                logger.info(
                    "Handoff %s reused for %s (%s, submitted %d times)",
                    record.handoff_id,
                    request.sender_masked,
                    request.kind.value,
                    record.submission_count,
                )
                return HandoffResult(
                    accepted=True,
                    outcome=HandoffOutcome.DEDUPLICATED,
                    handoff_id=record.handoff_id,
                    status=record.status,
                    reason="open_handoff_exists",
                    created_at=now,
                )

            self._sequence += 1
            handoff_id = f"ho-{self._sequence:06d}"
            # Own copy: the model is frozen, but its lists are not, and a
            # caller must not be able to edit a stored handoff after the fact.
            record = HandoffRecord(
                handoff_id=handoff_id, request=request.model_copy(deep=True), created_at=now, updated_at=now
            )
            self._records[handoff_id] = record
            self._open_by_key[request.dedupe_key] = handoff_id
            self._evict_if_needed()
            logger.info(
                "Handoff %s created for %s (%s, priority=%s, reasons=%s)",
                handoff_id,
                request.sender_masked,
                request.kind.value,
                request.priority.value,
                ",".join(request.reason_codes),
            )
            return HandoffResult(
                accepted=True,
                outcome=HandoffOutcome.CREATED,
                handoff_id=handoff_id,
                status=record.status,
                reason="created",
                created_at=now,
            )

    # -- Inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def get(self, handoff_id: str) -> Optional[HandoffRecord]:
        """A copy of the record, or ``None``."""
        with self._lock:
            record = self._records.get(handoff_id)
            return record.model_copy(deep=True) if record is not None else None

    def list(
        self,
        status: Optional[HandoffStatus] = None,
        conversation_id: Optional[str] = None,
    ) -> List[HandoffRecord]:
        """Copies of stored records, oldest first, optionally filtered."""
        with self._lock:
            return [
                record.model_copy(deep=True)
                for record in self._records.values()
                if (status is None or record.status == status)
                and (conversation_id is None or record.request.conversation_id == conversation_id)
            ]

    def open_handoff_for(self, request: HandoffRequest) -> Optional[str]:
        """The ID of the non-closed handoff ``request`` would dedupe into."""
        with self._lock:
            return self._open_by_key.get(request.dedupe_key)

    # -- Lifecycle ----------------------------------------------------------

    def accept(self, handoff_id: str) -> HandoffRecord:
        """``pending -> accepted``: a human has taken the handoff."""
        return self._transition(handoff_id, HandoffStatus.ACCEPTED, allowed_from=(HandoffStatus.PENDING,))

    def close(self, handoff_id: str) -> HandoffRecord:
        """``pending|accepted -> closed``. Frees the dedupe slot."""
        return self._transition(
            handoff_id, HandoffStatus.CLOSED, allowed_from=(HandoffStatus.PENDING, HandoffStatus.ACCEPTED)
        )

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._open_by_key.clear()

    # -- Internal -----------------------------------------------------------

    def _transition(self, handoff_id: str, target: HandoffStatus, allowed_from: tuple) -> HandoffRecord:
        with self._lock:
            record = self._records.get(handoff_id)
            if record is None:
                raise HandoffError(f"unknown handoff '{handoff_id}'", code="unknown_handoff")
            if record.status not in allowed_from:
                raise HandoffError(
                    f"cannot move handoff '{handoff_id}' from '{record.status.value}' to '{target.value}'",
                    code="invalid_transition",
                )
            record.status = target
            record.updated_at = self._clock()
            if target == HandoffStatus.CLOSED:
                self._open_by_key.pop(record.request.dedupe_key, None)
            return record.model_copy(deep=True)

    def _evict_if_needed(self) -> None:
        while len(self._records) > self.max_handoffs:
            victim_id = next(
                (hid for hid, rec in self._records.items() if rec.status == HandoffStatus.CLOSED),
                None,
            )
            if victim_id is None:
                victim_id = next(iter(self._records))
                logger.warning(
                    "Handoff sink at capacity (%d) with no closed handoffs; evicting oldest open handoff %s",
                    self.max_handoffs,
                    victim_id,
                )
            victim = self._records.pop(victim_id)
            if self._open_by_key.get(victim.request.dedupe_key) == victim_id:
                del self._open_by_key[victim.request.dedupe_key]

    def _rejected(self, reason: str) -> HandoffResult:
        return HandoffResult(
            accepted=False,
            outcome=HandoffOutcome.REJECTED,
            handoff_id=None,
            status=None,
            reason=reason,
            created_at=self._clock(),
        )


__all__ = [
    "DEFAULT_MAX_HANDOFFS",
    "HandoffError",
    "HandoffKind",
    "HandoffOutcome",
    "HandoffPriority",
    "HandoffRecord",
    "HandoffRequest",
    "HandoffResult",
    "HandoffSink",
    "HandoffSinkError",
    "HandoffStatus",
    "InMemoryHandoffSink",
    "LeadSnapshot",
    "MAX_SUMMARY_LENGTH",
    "MAX_TRANSCRIPT_ENTRIES",
    "MAX_TRANSCRIPT_MESSAGE_LENGTH",
    "TranscriptEntry",
    "conversation_id_for",
    "handoff_request_from_decision",
]
