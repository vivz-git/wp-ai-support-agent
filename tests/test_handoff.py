"""Tests for ``app.agent.handoff`` (Milestone 2, Slice 11).

The handoff boundary is exercised with real ``EscalationDecision`` and
``ConversationState`` objects plus an ``InMemoryHandoffSink``. No Groq, no
WhatsApp, no Meta, no network, no settings, and neither the adapter nor the
sink ever mutates the state or request it is given.
"""

import inspect
import json
import socket
from datetime import datetime, timedelta, timezone

import pytest

from app.agent import handoff as handoff_module
from app.agent.escalation import (
    EscalationAction,
    EscalationDecision,
    EscalationPolicy,
    UserMessageInstruction,
)
from app.agent.guardrails import (
    AngerResult,
    GroundingResult,
    GuardrailSignals,
    HumanRequestResult,
    InjectionResult,
    RepetitionResult,
    analyze_customer_message,
)
from app.agent.handoff import (
    DEFAULT_MAX_HANDOFFS,
    MAX_SUMMARY_LENGTH,
    MAX_TRANSCRIPT_ENTRIES,
    MAX_TRANSCRIPT_MESSAGE_LENGTH,
    HandoffError,
    HandoffKind,
    HandoffOutcome,
    HandoffPriority,
    HandoffRecord,
    HandoffRequest,
    HandoffResult,
    HandoffSink,
    HandoffStatus,
    InMemoryHandoffSink,
    LeadSnapshot,
    TranscriptEntry,
    conversation_id_for,
    handoff_request_from_decision,
)
from app.agent.lead import LeadDelta, QualificationState
from app.agent.state import MAX_CHAT_HISTORY, ConversationState, Intent, ToolInvocation

SENDER = "919876543210"
OTHER_SENDER = "919812345678"
SECRET = "sk-handoff-secret-9f8e7d6c"

WHOLESALE_DELTA = LeadDelta(
    track="wholesale",
    contact_name="Asha Rao",
    business_name="Third Wave Cafe",
    business_type="cafe",
    monthly_volume_kg=25,
    city="Bengaluru",
    timeline="within_1_month",
    email="Asha@ThirdWave.example",
    intent_summary="Wants a monthly espresso blend supply for two cafe outlets",
)

FIXED_NOW = datetime(2026, 9, 11, 10, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(sender: str = SENDER, turns: int = 1) -> ConversationState:
    state = ConversationState.new(sender)
    for _ in range(turns):
        state.begin_turn()
    return state


def _conversation(sender: str = SENDER) -> ConversationState:
    """A short realistic conversation with lead data and a tool call."""
    state = _state(sender)
    state.add_user_message("Hi, do you supply cafes? We need about 25kg a month in Bengaluru.")
    state.set_intent(Intent.WHOLESALE_INQUIRY, 0.9)
    state.record_tool_invocation(
        ToolInvocation(
            tool_name="lookup_product",
            turn=1,
            status="ok",
            ok=True,
            arguments={"query": "espresso"},
            result={"internal_sku": "ESP-INTERNAL-42", "price_inr": 850},
        )
    )
    state.add_assistant_message("Yes, we do wholesale. Could I get your business name and a timeline?")
    state.begin_turn()
    state.add_user_message("Third Wave Cafe, within a month. I'm Asha. Can I talk to a human?")
    state.apply_lead_delta(WHOLESALE_DELTA)
    return state


def _decision(action: EscalationAction, codes, priority: int) -> EscalationDecision:
    instruction = (
        UserMessageInstruction.OFFER_HUMAN_HANDOFF
        if action in (EscalationAction.ESCALATE, EscalationAction.HANDOFF_READY)
        else UserMessageInstruction.CONTINUE_NORMAL_RESPONSE
    )
    return EscalationDecision(
        action=action, reason_codes=list(codes), user_message_instruction=instruction, priority=priority
    )


def _escalate(codes=("human_requested",), priority: int = 3) -> EscalationDecision:
    return _decision(EscalationAction.ESCALATE, codes, priority)


def _handoff_ready(code: str = "lead_qualified") -> EscalationDecision:
    return _decision(EscalationAction.HANDOFF_READY, [code], 7)


def _request(sender: str = SENDER, codes=("human_requested",), state=None) -> HandoffRequest:
    request = handoff_request_from_decision(_escalate(codes), state or _conversation(sender), now=FIXED_NOW)
    assert request is not None
    return request


def _quiet_signals() -> GuardrailSignals:
    return GuardrailSignals(
        injection=InjectionResult(),
        anger=AngerResult(),
        repetition=RepetitionResult(),
        human_request=HumanRequestResult(),
        grounding=GroundingResult(),
    )


class _Clock:
    """Deterministic clock: each call advances by one second."""

    def __init__(self, start: datetime = FIXED_NOW):
        self.now = start

    def __call__(self) -> datetime:
        current = self.now
        self.now = self.now + timedelta(seconds=1)
        return current


@pytest.fixture
def sink() -> InMemoryHandoffSink:
    return InMemoryHandoffSink(clock=_Clock())


# ---------------------------------------------------------------------------
# 1-2. HandoffRequest construction and serialization
# ---------------------------------------------------------------------------


def test_create_handoff_request():
    request = _request()
    assert request.kind == HandoffKind.ESCALATION
    assert request.reason_codes == ["human_requested"]
    assert request.priority == HandoffPriority.HIGH
    assert request.policy_priority == 3
    assert request.conversation_id == conversation_id_for(SENDER)
    assert request.sender_masked == "********3210"
    assert request.turn == 2
    assert request.language == "en"
    assert request.created_at == FIXED_NOW
    assert isinstance(request.lead, LeadSnapshot)
    assert all(isinstance(entry, TranscriptEntry) for entry in request.transcript)


def test_request_is_immutable_and_strict():
    request = _request()
    with pytest.raises(Exception):
        request.priority = HandoffPriority.LOW  # type: ignore[misc]
    with pytest.raises(Exception):
        HandoffRequest(**request.model_dump(), extra_field="nope")
    with pytest.raises(Exception):
        HandoffRequest(**{**request.model_dump(), "reason_codes": []})


def test_request_serialization_round_trip():
    request = _request()
    payload = request.model_dump(mode="json")
    text = json.dumps(payload)  # JSON-compatible
    restored = HandoffRequest.model_validate(json.loads(text))
    assert restored == request
    assert payload["kind"] == "escalation"
    assert payload["priority"] == "high"
    assert payload["lead"]["business_name"] == "Third Wave Cafe"
    assert payload["created_at"].startswith("2026-09-11T10:00:00")


# ---------------------------------------------------------------------------
# 3-5. Lead snapshot safety, phone masking, no credentials
# ---------------------------------------------------------------------------


def test_safe_lead_snapshot_contains_useful_fields_only():
    request = _request()
    lead = request.lead
    assert lead.contact_name == "Asha Rao"
    assert lead.business_name == "Third Wave Cafe"
    assert lead.business_type == "cafe"
    assert lead.city == "Bengaluru"
    assert lead.email == "asha@thirdwave.example"
    assert lead.lead_track == "wholesale"
    assert lead.monthly_volume_kg == 25
    assert lead.timeline == "within_1_month"
    assert lead.intent_summary.startswith("Wants a monthly espresso")
    assert lead.qualification == "qualified"
    assert lead.missing_required_fields == []
    dumped = lead.model_dump()
    for forbidden in ("whatsapp_number", "field_provenance", "source", "prompt", "system"):
        assert forbidden not in dumped


def test_no_full_phone_number_anywhere_in_request():
    state = _conversation()
    state.add_assistant_message(f"Thanks, I have your number as {SENDER}.")
    state.add_user_message(f"Yes {SENDER} is right, call me")
    state.lead.intent_summary = f"Call back on {SENDER} about beans"
    request = handoff_request_from_decision(_escalate(), state)
    dumped = request.model_dump_json()
    assert SENDER not in dumped
    assert "********3210" in dumped
    # The number was in the transcript and the lead text; both are masked, not dropped.
    assert any("********3210 is right" in entry.content for entry in request.transcript)
    assert request.lead.intent_summary == "Call back on ********3210 about beans"


def test_no_credentials_in_request(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAAB-whatsapp-token-xyz")
    request = _request()
    dumped = request.model_dump_json()
    assert SECRET not in dumped
    assert "EAAB-whatsapp-token-xyz" not in dumped
    for forbidden in ("api_key", "access_token", "Authorization", "Bearer", "GROQ", "system_prompt"):
        assert forbidden not in dumped


# ---------------------------------------------------------------------------
# 6-8, 30. Conversation context: bounded, no system messages, no tools
# ---------------------------------------------------------------------------


def test_bounded_conversation_context():
    state = _state(turns=1)
    for i in range(MAX_CHAT_HISTORY * 3):
        state.add_user_message(f"message {i}")
    request = handoff_request_from_decision(_escalate(), state)
    assert len(request.transcript) == MAX_TRANSCRIPT_ENTRIES == MAX_CHAT_HISTORY
    # Most recent messages are kept, oldest dropped.
    assert request.transcript[-1].content == f"message {MAX_CHAT_HISTORY * 3 - 1}"
    assert request.transcript[0].content == f"message {MAX_CHAT_HISTORY * 2}"


def test_long_history_and_long_messages_bounded():
    state = _state(turns=1)
    state.add_user_message("x" * 4096)
    state.add_assistant_message("y" * 4096)
    request = handoff_request_from_decision(_escalate(), state)
    for entry in request.transcript:
        assert len(entry.content) == MAX_TRANSCRIPT_MESSAGE_LENGTH
        assert entry.content.endswith("…")
    assert len(request.summary) <= MAX_SUMMARY_LENGTH


def test_system_messages_excluded():
    """State history only ever holds user/assistant turns; the transcript
    re-labels them and never introduces a system role."""
    request = _request()
    roles = {entry.role for entry in request.transcript}
    assert roles == {"customer", "assistant"}
    assert "system" not in request.model_dump_json()
    assert TranscriptEntry.model_fields["role"].annotation.__args__ == ("customer", "assistant")
    with pytest.raises(Exception):
        TranscriptEntry(role="system", content="You are a helpful assistant", turn=0)


def test_tool_internals_excluded():
    state = _conversation()
    assert state.tool_history and state.current_turn_tool_results is not None
    request = handoff_request_from_decision(_escalate(), state)
    dumped = request.model_dump_json()
    assert "lookup_product" not in dumped
    assert "ESP-INTERNAL-42" not in dumped
    assert "tool_history" not in dumped
    assert "arguments" not in dumped
    assert "known_facts" not in dumped
    assert "flags" not in dumped


# ---------------------------------------------------------------------------
# 9-13, 32-33. Adapter semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action, codes, priority",
    [
        (EscalationAction.CONTINUE, ["no_rule_fired"], 12),
        (EscalationAction.CLARIFY, ["repeated_question"], 11),
        (EscalationAction.REFUSE, ["injection_attempt"], 9),
        (EscalationAction.SUPPRESS, ["grounding_violation"], 1),
    ],
)
def test_non_escalation_decision_produces_no_request(action, codes, priority):
    decision = _decision(action, codes, priority)
    assert handoff_request_from_decision(decision, _conversation()) is None


def test_escalation_decision_produces_request():
    request = handoff_request_from_decision(_escalate(["human_requested"], 3), _conversation())
    assert request is not None
    assert request.kind == HandoffKind.ESCALATION
    assert request.summary.startswith("Escalation for ********3210 after 2 turns (human_requested).")
    assert "Wholesale lead: Asha Rao, Third Wave Cafe, cafe, Bengaluru." in request.summary


def test_handoff_ready_represented_safely():
    state = _conversation()
    assert state.qualification == QualificationState.QUALIFIED
    # Not yet consented: the policy emits ``lead_qualified``.
    request = handoff_request_from_decision(_handoff_ready("lead_qualified"), state)
    assert request.kind == HandoffKind.QUALIFIED_LEAD
    assert request.priority == HandoffPriority.LOW
    assert request.lead.qualification == "qualified"
    assert request.summary.startswith("Qualified lead for ********3210")
    # Consented: the state is handoff_ready and the code changes accordingly.
    state.mark_handoff_ready()
    consented = handoff_request_from_decision(_handoff_ready("handoff_ready"), state)
    assert consented.kind == HandoffKind.QUALIFIED_LEAD
    assert consented.priority == HandoffPriority.MEDIUM
    assert consented.lead.qualification == "handoff_ready"
    assert consented.dedupe_key == request.dedupe_key  # same opportunity, one ticket


@pytest.mark.parametrize(
    "codes, policy_priority, expected",
    [
        (["human_requested"], 3, HandoffPriority.HIGH),
        (["high_anger_complaint"], 4, HandoffPriority.URGENT),
        (["repeated_unresolved"], 5, HandoffPriority.HIGH),
        (["injection_repeated", "injection_attempt"], 6, HandoffPriority.MEDIUM),
        (["already_escalated"], 2, HandoffPriority.MEDIUM),
        (["already_escalated", "high_anger_complaint"], 2, HandoffPriority.URGENT),
        (["some_future_rule"], 3, HandoffPriority.MEDIUM),
    ],
)
def test_priority_preserved_and_derived(codes, policy_priority, expected):
    request = handoff_request_from_decision(_escalate(codes, policy_priority), _conversation())
    assert request.policy_priority == policy_priority
    assert request.priority == expected


def test_reason_codes_preserved_in_order():
    codes = ["already_escalated", "human_requested", "repeated_unresolved"]
    request = handoff_request_from_decision(_escalate(codes, 2), _conversation())
    assert request.reason_codes == codes
    assert request.primary_reason == "already_escalated"
    assert "already_escalated, human_requested, repeated_unresolved" in request.summary


def test_adapter_preserves_current_escalation_semantics():
    """Drive the real policy end-to-end: the adapter mirrors its verdicts."""
    policy = EscalationPolicy()
    quiet = _state()
    assert handoff_request_from_decision(policy.evaluate(_quiet_signals(), quiet), quiet) is None

    asking = _state()
    signals = analyze_customer_message("I want to talk to a human!!!", [])
    decision = policy.evaluate(signals, asking)
    request = handoff_request_from_decision(decision, asking)
    assert decision.action == EscalationAction.ESCALATE
    assert request.reason_codes == decision.reason_codes == ["human_requested"]
    assert request.policy_priority == decision.priority == 3

    sticky = _state()
    sticky.mark_escalated("human_requested")
    sticky_decision = policy.evaluate(_quiet_signals(), sticky)
    sticky_request = handoff_request_from_decision(sticky_decision, sticky)
    assert sticky_request.reason_codes == ["already_escalated"]
    assert sticky_request.dedupe_key == request.dedupe_key

    qualified = _conversation()
    qualified_decision = policy.evaluate(_quiet_signals(), qualified)
    assert qualified_decision.action == EscalationAction.HANDOFF_READY
    assert handoff_request_from_decision(qualified_decision, qualified).kind == HandoffKind.QUALIFIED_LEAD


def test_non_escalation_actions_never_generate_handoffs(sink):
    for action in EscalationAction:
        if action in (EscalationAction.ESCALATE, EscalationAction.HANDOFF_READY):
            continue
        request = handoff_request_from_decision(_decision(action, ["x"], 9), _conversation())
        assert request is None
    assert len(sink) == 0


def test_adapter_does_not_mutate_state():
    state = _conversation()
    before = state.model_dump(mode="json")
    handoff_request_from_decision(_escalate(), state)
    handoff_request_from_decision(_handoff_ready(), state)
    assert state.model_dump(mode="json") == before


# ---------------------------------------------------------------------------
# 14-17, 35. Sink accepts, deterministic IDs, dedupe
# ---------------------------------------------------------------------------


def test_in_memory_sink_accepts_request(sink):
    request = _request()
    result = sink.submit(request)
    assert isinstance(result, HandoffResult)
    assert result.accepted is True
    assert result.outcome == HandoffOutcome.CREATED
    assert result.status == HandoffStatus.PENDING
    assert result.reason == "created"
    assert result.handoff_id == "ho-000001"
    assert result.created_at == FIXED_NOW
    assert len(sink) == 1
    assert isinstance(sink, HandoffSink)


def test_handoff_id_deterministic_across_fresh_sinks():
    requests = [_request(SENDER), _request(OTHER_SENDER)]
    sink_a = InMemoryHandoffSink(clock=_Clock())
    sink_b = InMemoryHandoffSink(clock=_Clock())
    ids_a = [sink_a.submit(r).handoff_id for r in requests]
    ids_b = [sink_b.submit(r).handoff_id for r in requests]
    assert ids_a == ids_b == ["ho-000001", "ho-000002"]
    # A brand-new sink starts the sequence again: no hidden global counter.
    assert InMemoryHandoffSink().submit(requests[1]).handoff_id == "ho-000001"


def test_repeated_same_escalation_deduplicates(sink):
    first = sink.submit(_request())
    second = sink.submit(_request())
    third = sink.submit(_request(codes=("already_escalated",)))
    assert first.outcome == HandoffOutcome.CREATED
    assert second.outcome == third.outcome == HandoffOutcome.DEDUPLICATED
    assert second.accepted and third.accepted
    assert second.handoff_id == third.handoff_id == first.handoff_id
    assert second.reason == "open_handoff_exists"
    assert len(sink) == 1
    record = sink.get(first.handoff_id)
    assert record.submission_count == 3
    assert record.request.reason_codes == ["human_requested"]  # original kept


def test_different_escalation_reasons_behave_deterministically(sink):
    human = sink.submit(_request(codes=("human_requested",)))
    angry = sink.submit(_request(codes=("high_anger_complaint",)))
    # Same active escalation for the same conversation: one ticket.
    assert angry.outcome == HandoffOutcome.DEDUPLICATED
    assert angry.handoff_id == human.handoff_id
    # A different kind for the same conversation is a separate handoff.
    lead = sink.submit(handoff_request_from_decision(_handoff_ready(), _conversation()))
    assert lead.outcome == HandoffOutcome.CREATED
    assert lead.handoff_id == "ho-000002"
    # Once closed, the same escalation reopens as a fresh handoff.
    sink.close(human.handoff_id)
    reopened = sink.submit(_request(codes=("high_anger_complaint",)))
    assert reopened.outcome == HandoffOutcome.CREATED
    assert reopened.handoff_id == "ho-000003"
    assert len(sink) == 3


def test_deterministic_repeated_submission():
    def run() -> list:
        sink = InMemoryHandoffSink(clock=_Clock())
        return [
            sink.submit(_request(SENDER)).model_dump(mode="json"),
            sink.submit(_request(SENDER)).model_dump(mode="json"),
            sink.submit(_request(OTHER_SENDER)).model_dump(mode="json"),
        ]

    assert run() == run()


# ---------------------------------------------------------------------------
# 18-20. get / list / close (+ accept)
# ---------------------------------------------------------------------------


def test_get_handoff(sink):
    result = sink.submit(_request())
    record = sink.get(result.handoff_id)
    assert isinstance(record, HandoffRecord)
    assert record.handoff_id == result.handoff_id
    assert record.status == HandoffStatus.PENDING
    assert record.request.sender_masked == "********3210"
    assert record.created_at == record.updated_at == FIXED_NOW
    assert sink.get("ho-999999") is None
    # Copies, not references.
    record.status = HandoffStatus.CLOSED
    assert sink.get(result.handoff_id).status == HandoffStatus.PENDING


def test_list_handoffs(sink):
    a = sink.submit(_request(SENDER)).handoff_id
    b = sink.submit(_request(OTHER_SENDER)).handoff_id
    c = sink.submit(handoff_request_from_decision(_handoff_ready(), _conversation(SENDER))).handoff_id
    assert [r.handoff_id for r in sink.list()] == [a, b, c]
    sink.close(b)
    assert [r.handoff_id for r in sink.list(status=HandoffStatus.PENDING)] == [a, c]
    assert [r.handoff_id for r in sink.list(status=HandoffStatus.CLOSED)] == [b]
    assert [r.handoff_id for r in sink.list(conversation_id=conversation_id_for(SENDER))] == [a, c]
    assert sink.list(conversation_id="conv_nobody") == []


def test_close_handoff(sink):
    handoff_id = sink.submit(_request()).handoff_id
    accepted = sink.accept(handoff_id)
    assert accepted.status == HandoffStatus.ACCEPTED
    assert accepted.updated_at > accepted.created_at
    closed = sink.close(handoff_id)
    assert closed.status == HandoffStatus.CLOSED
    assert sink.get(handoff_id).status == HandoffStatus.CLOSED
    assert sink.open_handoff_for(_request()) is None
    with pytest.raises(HandoffError) as excinfo:
        sink.close(handoff_id)
    assert excinfo.value.code == "invalid_transition"
    with pytest.raises(HandoffError) as excinfo:
        sink.accept(handoff_id)
    assert excinfo.value.code == "invalid_transition"
    with pytest.raises(HandoffError) as excinfo:
        sink.close("ho-424242")
    assert excinfo.value.code == "unknown_handoff"


# ---------------------------------------------------------------------------
# 21-22. Bounded storage and deterministic eviction
# ---------------------------------------------------------------------------


def _sender(i: int) -> str:
    return f"9198{i:08d}"


def test_bounded_storage():
    assert DEFAULT_MAX_HANDOFFS == 500
    sink = InMemoryHandoffSink(max_handoffs=5, clock=_Clock())
    for i in range(20):
        sink.submit(_request(_sender(i)))
    assert len(sink) == 5
    with pytest.raises(ValueError):
        InMemoryHandoffSink(max_handoffs=0)


def test_oldest_closed_evicted_first_deterministically():
    sink = InMemoryHandoffSink(max_handoffs=3, clock=_Clock())
    ids = [sink.submit(_request(_sender(i))).handoff_id for i in range(3)]
    sink.close(ids[1])  # ho-000002 closed; ho-000001 is older but open
    sink.submit(_request(_sender(3)))
    remaining = [r.handoff_id for r in sink.list()]
    assert remaining == [ids[0], ids[2], "ho-000004"]
    # No closed entries left: the oldest open one goes, and its dedupe slot with it.
    sink.submit(_request(_sender(4)))
    assert [r.handoff_id for r in sink.list()] == [ids[2], "ho-000004", "ho-000005"]
    assert sink.open_handoff_for(_request(_sender(0))) is None
    # Resubmitting the evicted conversation creates a fresh handoff, not a dangling dedupe.
    assert sink.submit(_request(_sender(0))).outcome == HandoffOutcome.CREATED


# ---------------------------------------------------------------------------
# 23-25. Sink never mutates inputs; malformed input rejected safely
# ---------------------------------------------------------------------------


def test_sink_does_not_mutate_request(sink):
    request = _request()
    before = request.model_dump(mode="json")
    sink.submit(request)
    sink.submit(request)
    assert request.model_dump(mode="json") == before
    # And the stored copy is independent of the caller's object.
    request.reason_codes.append("tampered")
    assert sink.get("ho-000001").request.reason_codes == ["human_requested"]


def test_sink_does_not_mutate_conversation_state(sink):
    state = _conversation()
    before = state.model_dump(mode="json")
    request = handoff_request_from_decision(_escalate(), state)
    sink.submit(request)
    sink.close("ho-000001")
    assert state.model_dump(mode="json") == before
    assert state.escalation.status.value == "none"  # the sink never applies transitions


def test_malformed_request_rejected_safely(sink):
    for bad in (None, {}, "handoff me", 42, _conversation()):
        result = sink.submit(bad)  # type: ignore[arg-type]
        assert result.accepted is False
        assert result.outcome == HandoffOutcome.REJECTED
        assert result.handoff_id is None
        assert result.status is None
        assert result.reason == "malformed_request"
    assert len(sink) == 0
    # A structurally valid instance whose list was tampered past validation is also rejected.
    request = _request()
    request.reason_codes.clear()
    assert sink.submit(request).reason == "invalid_request"
    assert len(sink) == 0


# ---------------------------------------------------------------------------
# 26-28. No network, no env/API-key access, sender isolation
# ---------------------------------------------------------------------------


def test_no_network_access(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during handoff")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    sink = InMemoryHandoffSink()
    result = sink.submit(_request())
    assert result.accepted
    assert sink.get(result.handoff_id) is not None
    sink.close(result.handoff_id)


def test_no_api_key_or_env_access(monkeypatch):
    source = inspect.getsource(handoff_module)
    for forbidden in (
        "os.environ",
        "getenv",
        "get_settings",
        "Settings",
        "dotenv",
        "import httpx",
        "import requests",
        "import aiohttp",
        "import urllib",
    ):
        assert forbidden not in source
    assert "from app.config import mask_phone_number" in source
    # Even with secrets present in the environment, nothing reaches the payload.
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "EAAB-token-abc")
    sink = InMemoryHandoffSink()
    result = sink.submit(_request())
    dumped = result.model_dump_json() + sink.get(result.handoff_id).model_dump_json()
    assert SECRET not in dumped
    assert "EAAB-token-abc" not in dumped


def test_sender_isolation(sink):
    a = sink.submit(_request(SENDER))
    b = sink.submit(_request(OTHER_SENDER))
    assert a.handoff_id != b.handoff_id
    assert conversation_id_for(SENDER) != conversation_id_for(OTHER_SENDER)
    a_record, b_record = sink.get(a.handoff_id), sink.get(b.handoff_id)
    assert a_record.request.conversation_id != b_record.request.conversation_id
    assert a_record.request.sender_masked == "********3210"
    assert b_record.request.sender_masked == "********5678"
    # Closing one sender's handoff does not touch the other's dedupe slot.
    sink.close(a.handoff_id)
    assert sink.submit(_request(OTHER_SENDER)).outcome == HandoffOutcome.DEDUPLICATED
    assert sink.submit(_request(SENDER)).outcome == HandoffOutcome.CREATED
    # A masked collision (same last four digits) is still two conversations.
    twin = "911112223210"
    assert conversation_id_for(twin) != conversation_id_for(SENDER)
    assert sink.submit(_request(twin)).outcome == HandoffOutcome.CREATED


# ---------------------------------------------------------------------------
# 29, 36. Empty lead / minimal state
# ---------------------------------------------------------------------------


def test_empty_lead_handled():
    state = _state()
    state.add_user_message("get me a person")
    request = handoff_request_from_decision(_escalate(), state)
    assert request.lead.is_empty()
    assert request.lead.lead_track == "unknown"
    assert request.lead.qualification == "unknown"
    assert request.lead.missing_required_fields == ["contact_name", "brew_method", "taste_preference"]
    assert "Lead track: unknown; qualification: unknown." in request.summary
    assert SENDER not in request.model_dump_json()


def test_safe_behavior_with_minimal_state(sink):
    state = ConversationState.new("agent-test-user")  # no turns, no history, non-numeric sender
    request = handoff_request_from_decision(_escalate(["already_escalated"], 2), state)
    assert request is not None
    assert request.turn == 0
    assert request.transcript == []
    assert request.sender_masked == "***********user"
    assert request.summary.startswith("Escalation for ***********user after 0 turns")
    assert request.lead.is_empty()
    result = sink.submit(request)
    assert result.accepted and result.outcome == HandoffOutcome.CREATED


# ---------------------------------------------------------------------------
# 31, 34. Serialization has no secrets; no global singleton
# ---------------------------------------------------------------------------


def test_serialization_contains_no_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", SECRET)
    state = _conversation()
    state.remember_fact("internal_note", "Bearer super-secret-internal-token")
    sink = InMemoryHandoffSink(clock=_Clock())
    request = handoff_request_from_decision(_escalate(), state)
    result = sink.submit(request)
    everything = json.dumps(
        {
            "request": request.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "records": [r.model_dump(mode="json") for r in sink.list()],
        }
    )
    assert SECRET not in everything
    assert "super-secret-internal-token" not in everything
    assert "Bearer" not in everything
    assert SENDER not in everything
    assert "ESP-INTERNAL-42" not in everything
    assert "mock_test_access_token" not in everything  # conftest's WhatsApp token
    assert "mock_groq_api_key" not in everything


def test_no_global_singleton():
    for name, value in vars(handoff_module).items():
        assert not isinstance(value, InMemoryHandoffSink), f"module-level sink instance: {name}"
    assert "default_sink" not in vars(handoff_module)
    assert "get_handoff_sink" not in vars(handoff_module)
    # Two sinks are fully independent.
    a, b = InMemoryHandoffSink(), InMemoryHandoffSink()
    a.submit(_request())
    assert len(a) == 1 and len(b) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_result_is_provider_neutral():
    fields = set(HandoffResult.model_fields)
    assert fields == {"accepted", "outcome", "handoff_id", "status", "reason", "created_at"}
    with pytest.raises(Exception):
        HandoffResult(accepted=True, outcome="created", reason="created", slack_ts="123")


def test_handoff_ready_and_escalation_are_separate_tickets_but_each_deduped(sink):
    lead_state = _conversation()
    lead_a = sink.submit(handoff_request_from_decision(_handoff_ready(), lead_state))
    lead_b = sink.submit(handoff_request_from_decision(_handoff_ready(), lead_state))
    escalation = sink.submit(handoff_request_from_decision(_escalate(), lead_state))
    assert lead_a.outcome == HandoffOutcome.CREATED
    assert lead_b.outcome == HandoffOutcome.DEDUPLICATED and lead_b.handoff_id == lead_a.handoff_id
    assert escalation.outcome == HandoffOutcome.CREATED and escalation.handoff_id != lead_a.handoff_id


def test_clear_resets_records_but_not_id_sequence(sink):
    sink.submit(_request())
    sink.clear()
    assert len(sink) == 0
    assert sink.submit(_request()).handoff_id == "ho-000002"  # ids are never reused
