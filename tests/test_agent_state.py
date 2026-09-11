"""Tests for the conversation state model and store (app/agent/state.py,
app/agent/store.py).

Self-contained: does not depend on tests/conftest.py's WhatsApp/Groq mocks.
Everything here is pure data; no network, no LLM, no database.
"""

import json
import socket

import pytest
from pydantic import ValidationError

from app.agent import (
    ConversationFlags,
    ConversationState,
    ConversationStore,
    EscalationState,
    EscalationStatus,
    Intent,
    InvalidTransitionError,
    LeadDelta,
    LeadTrack,
    QualificationState,
    ToolInvocation,
)
from app.agent.state import (
    MAX_CHAT_HISTORY,
    MAX_INTENT_HISTORY,
    MAX_KNOWN_FACTS,
    MAX_TOOL_HISTORY,
    STORE_CAPACITY,
    HistoryMessage,
)
from app.llm.base import ChatMessage

SENDER = "919876543210"

WHOLESALE_DELTA = LeadDelta(
    track="wholesale",
    contact_name="Asha Rao",
    business_name="Third Wave Cafe",
    business_type="cafe",
    monthly_volume_kg=25,
    city="Bengaluru",
    timeline="within_1_month",
)


def _qualified_state() -> ConversationState:
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.apply_lead_delta(WHOLESALE_DELTA)
    assert state.qualification == QualificationState.QUALIFIED
    return state


# ---------------------------------------------------------------------------
# 1. Enum values
# ---------------------------------------------------------------------------


def test_intent_enum_values():
    assert [i.value for i in Intent] == [
        "faq",
        "product_inquiry",
        "recommendation",
        "price_check",
        "availability_check",
        "wholesale_inquiry",
        "order_status",
        "complaint",
        "human_request",
        "smalltalk",
        "out_of_scope",
        "unclear",
    ]


def test_qualification_state_enum_values():
    assert [q.value for q in QualificationState] == [
        "unknown",
        "browsing",
        "collecting",
        "qualified",
        "handoff_ready",
        "declined",
        "escalated",
    ]


def test_escalation_status_enum_values():
    assert [e.value for e in EscalationStatus] == ["none", "pending", "handed_off"]


def test_lead_track_enum_values():
    assert [t.value for t in LeadTrack] == ["unknown", "consumer", "wholesale"]


def test_enums_are_string_valued():
    assert Intent("faq") is Intent.FAQ
    assert QualificationState("qualified") is QualificationState.QUALIFIED
    assert isinstance(Intent.FAQ, str)


# ---------------------------------------------------------------------------
# 2. Default ConversationState
# ---------------------------------------------------------------------------


def test_default_conversation_state():
    state = ConversationState.new(SENDER)
    assert state.sender_id == SENDER
    assert state.turn_count == 0
    assert state.language == "en"
    assert state.created_at <= state.updated_at
    assert state.current_intent == Intent.UNCLEAR
    assert state.intent_confidence == 0.0
    assert state.intent_history == []
    assert state.qualification == QualificationState.UNKNOWN
    assert state.known_facts == {}
    assert state.history == []
    assert state.tool_history == []
    assert state.current_turn_tool_results == []
    assert state.escalation == EscalationState()
    assert state.escalation.status == EscalationStatus.NONE
    assert state.flags == ConversationFlags()
    assert state.is_terminal is False


def test_new_state_stores_whatsapp_number_as_metadata_without_asking():
    state = ConversationState.new(SENDER)
    assert state.lead.whatsapp_number == SENDER
    # Not extracted from a turn, so it has no provenance entry.
    assert "whatsapp_number" not in state.lead.field_provenance
    assert state.lead.has_any_lead_data() is False


def test_new_state_with_non_phone_sender_leaves_number_unset():
    state = ConversationState.new("test-sender")
    assert state.lead.whatsapp_number is None


def test_state_rejects_blank_sender_and_unknown_fields():
    with pytest.raises(ValidationError):
        ConversationState(sender_id="   ")
    with pytest.raises(ValidationError):
        ConversationState(sender_id=SENDER, bogus=1)


# ---------------------------------------------------------------------------
# 3. State serialization round-trip
# ---------------------------------------------------------------------------


def test_state_serialization_round_trip():
    state = _qualified_state()
    state.set_intent(Intent.WHOLESALE_INQUIRY, 0.93)
    state.add_user_message("We run a cafe and need about 25kg a month")
    state.add_assistant_message("Great — let me note that down.")
    state.record_tool_invocation(
        ToolInvocation(tool_name="product_lookup", turn=1, status="ok", ok=True, arguments={"query": "espresso"}, result={"status": "ok"})
    )
    state.remember_fact("preferred_roast", "medium_dark")
    state.flags.anger_score = 0.2

    payload = state.model_dump(mode="json")
    text = json.dumps(payload)  # must be plain-JSON serializable
    restored = ConversationState.model_validate(json.loads(text))

    assert restored == state
    assert restored.model_dump(mode="json") == payload
    assert restored.lead.track == LeadTrack.WHOLESALE
    assert restored.qualification == QualificationState.QUALIFIED
    assert restored.current_turn_tool_results[0].result == {"status": "ok"}
    assert restored.tool_history[0].result is None


def test_round_trip_rejects_corrupt_payload():
    payload = ConversationState.new(SENDER).model_dump(mode="json")
    payload["qualification"] = "definitely_qualified"
    with pytest.raises(ValidationError):
        ConversationState.model_validate(payload)


# ---------------------------------------------------------------------------
# 4. Intent updates
# ---------------------------------------------------------------------------


def test_set_intent_updates_current_and_history():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    record = state.set_intent(Intent.PRICE_CHECK, 0.8)
    assert state.current_intent == Intent.PRICE_CHECK
    assert state.intent_confidence == 0.8
    assert state.intent_history == [record]
    assert record.turn == 1

    state.begin_turn()
    state.set_intent(Intent.HUMAN_REQUEST, 1.0)
    assert state.current_intent == Intent.HUMAN_REQUEST
    assert [r.intent for r in state.intent_history] == [Intent.PRICE_CHECK, Intent.HUMAN_REQUEST]
    assert state.intent_history[-1].turn == 2


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_set_intent_rejects_out_of_range_confidence(bad_confidence):
    state = ConversationState.new(SENDER)
    with pytest.raises(ValidationError):
        state.set_intent(Intent.FAQ, bad_confidence)


def test_intent_must_be_a_legal_enum_value():
    state = ConversationState.new(SENDER)
    with pytest.raises(ValidationError):
        state.current_intent = "buy_now"


# ---------------------------------------------------------------------------
# 5. Bounded intent history
# ---------------------------------------------------------------------------


def test_intent_history_is_bounded_and_keeps_newest():
    state = ConversationState.new(SENDER)
    for i in range(MAX_INTENT_HISTORY + 5):
        state.begin_turn()
        state.set_intent(Intent.FAQ, 0.5)
    assert len(state.intent_history) == MAX_INTENT_HISTORY
    assert state.intent_history[0].turn == 6
    assert state.intent_history[-1].turn == MAX_INTENT_HISTORY + 5


def test_intent_history_is_bounded_on_load():
    payload = ConversationState.new(SENDER).model_dump(mode="json")
    payload["intent_history"] = [
        {"intent": "faq", "confidence": 0.5, "turn": i} for i in range(MAX_INTENT_HISTORY + 10)
    ]
    restored = ConversationState.model_validate(payload)
    assert len(restored.intent_history) == MAX_INTENT_HISTORY
    assert restored.intent_history[-1].turn == MAX_INTENT_HISTORY + 9


# ---------------------------------------------------------------------------
# 6. Bounded chat history
# ---------------------------------------------------------------------------


def test_chat_history_is_bounded_and_ordered():
    state = ConversationState.new(SENDER)
    for i in range(MAX_CHAT_HISTORY + 4):
        state.begin_turn()
        state.add_user_message(f"user {i}")
    assert len(state.history) == MAX_CHAT_HISTORY
    assert state.history[0].content == "user 4"
    assert state.history[-1].content == f"user {MAX_CHAT_HISTORY + 3}"
    assert all(isinstance(m, HistoryMessage) for m in state.history)


def test_chat_messages_view_matches_llm_type():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.add_user_message("hi")
    state.add_assistant_message("hello")
    messages = state.chat_messages()
    assert messages == [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")]


def test_chat_history_rejects_empty_and_over_long_messages():
    state = ConversationState.new(SENDER)
    with pytest.raises(ValidationError):
        state.add_user_message("")
    with pytest.raises(ValidationError):
        state.add_assistant_message("x" * 4097)


def test_tool_history_is_bounded_and_current_turn_resets():
    state = ConversationState.new(SENDER)
    for i in range(MAX_TOOL_HISTORY + 3):
        state.begin_turn()
        state.record_tool_invocation(ToolInvocation(tool_name="product_lookup", turn=i, status="ok", ok=True))
    assert len(state.tool_history) == MAX_TOOL_HISTORY
    assert len(state.current_turn_tool_results) == 1
    state.begin_turn()
    assert state.current_turn_tool_results == []


def test_tool_failures_this_turn_counts_and_resets():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.record_tool_invocation(ToolInvocation(tool_name="product_lookup", turn=1, status="unavailable", ok=False))
    state.record_tool_invocation(ToolInvocation(tool_name="product_lookup", turn=1, status="ok", ok=True))
    assert state.flags.tool_failures_this_turn == 1
    state.begin_turn()
    assert state.flags.tool_failures_this_turn == 0


def test_known_facts_are_bounded():
    state = ConversationState.new(SENDER)
    for i in range(MAX_KNOWN_FACTS + 5):
        state.remember_fact(f"fact_{i}", "yes")
    assert len(state.known_facts) == MAX_KNOWN_FACTS
    assert "fact_0" not in state.known_facts
    assert f"fact_{MAX_KNOWN_FACTS + 4}" in state.known_facts

    with pytest.raises(ValidationError):
        state.known_facts = {f"k{i}": "v" for i in range(MAX_KNOWN_FACTS + 1)}


# ---------------------------------------------------------------------------
# 22. Transition to collecting
# ---------------------------------------------------------------------------


def test_transition_unknown_to_browsing_to_collecting():
    state = ConversationState.new(SENDER)
    assert state.reevaluate_qualification() == QualificationState.UNKNOWN

    state.begin_turn()
    assert state.reevaluate_qualification() == QualificationState.BROWSING

    state.apply_lead_delta(LeadDelta(contact_name="Asha"))
    assert state.qualification == QualificationState.COLLECTING
    assert state.lead.field_provenance == {"contact_name": 1}


def test_apply_lead_delta_uses_current_turn_for_provenance():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.begin_turn()
    state.begin_turn()
    state.apply_lead_delta(LeadDelta(city="Pune"))
    assert state.lead.field_provenance == {"city": 3}


# ---------------------------------------------------------------------------
# 23. Transition to qualified
# ---------------------------------------------------------------------------


def test_transition_collecting_to_qualified_across_turns():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.apply_lead_delta(LeadDelta(track="wholesale", contact_name="Asha Rao", business_name="Third Wave Cafe"))
    assert state.qualification == QualificationState.COLLECTING

    state.begin_turn()
    state.apply_lead_delta(LeadDelta(business_type="cafe", monthly_volume_kg=25))
    assert state.qualification == QualificationState.COLLECTING

    state.begin_turn()
    result = state.apply_lead_delta(LeadDelta(city="Bengaluru", timeline="immediate"))
    assert result == QualificationState.QUALIFIED
    assert state.qualification == QualificationState.QUALIFIED
    assert state.lead.field_provenance["timeline"] == 3
    assert state.is_terminal is False


def test_qualification_cannot_be_set_by_a_delta():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    with pytest.raises(ValidationError):
        LeadDelta(qualification="qualified")
    # A bare-minimum delta still leaves the state unqualified.
    state.apply_lead_delta(LeadDelta(contact_name="Asha"))
    assert state.qualification == QualificationState.COLLECTING


def test_direct_assignment_of_qualified_is_recomputed_on_reevaluation():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.qualification = QualificationState.QUALIFIED  # bypass attempt
    assert state.reevaluate_qualification() == QualificationState.BROWSING


# ---------------------------------------------------------------------------
# 24. Transition to handoff_ready
# ---------------------------------------------------------------------------


def test_transition_qualified_to_handoff_ready():
    state = _qualified_state()
    assert state.mark_handoff_ready() == QualificationState.HANDOFF_READY
    assert state.is_terminal is True
    # Sticky: re-evaluation does not pull it back to qualified.
    assert state.reevaluate_qualification() == QualificationState.HANDOFF_READY


def test_handoff_ready_requires_qualified():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.apply_lead_delta(LeadDelta(contact_name="Asha"))
    with pytest.raises(InvalidTransitionError):
        state.mark_handoff_ready()
    assert state.qualification == QualificationState.COLLECTING


def test_handoff_ready_rejects_forged_qualified_state():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.qualification = QualificationState.QUALIFIED  # forged, profile empty
    with pytest.raises(InvalidTransitionError):
        state.mark_handoff_ready()


def test_handed_off_from_handoff_ready():
    state = _qualified_state()
    state.mark_handoff_ready()
    state.begin_turn()
    assert state.mark_handed_off() == EscalationStatus.HANDED_OFF
    assert state.escalation.handed_off_at_turn == 2


def test_handed_off_requires_pending_or_handoff_ready():
    state = ConversationState.new(SENDER)
    with pytest.raises(InvalidTransitionError):
        state.mark_handed_off()


# ---------------------------------------------------------------------------
# 25. Declined state
# ---------------------------------------------------------------------------


def test_declined_state():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.apply_lead_delta(LeadDelta(contact_name="Asha"))
    assert state.mark_declined() == QualificationState.DECLINED
    assert state.flags.declines == 1
    assert state.is_terminal is True

    # Sticky: more data does not silently re-qualify a declined lead.
    state.begin_turn()
    state.apply_lead_delta(WHOLESALE_DELTA)
    assert state.qualification == QualificationState.DECLINED
    assert state.lead.business_name == "Third Wave Cafe"  # data still recorded


def test_declined_from_qualified():
    state = _qualified_state()
    assert state.mark_declined() == QualificationState.DECLINED


def test_cannot_decline_escalated_conversation():
    state = ConversationState.new(SENDER)
    state.mark_escalated("angry customer")
    with pytest.raises(InvalidTransitionError):
        state.mark_declined()


# ---------------------------------------------------------------------------
# 26. Escalated state
# ---------------------------------------------------------------------------


def test_escalated_state_from_any_state():
    for setup in (
        lambda s: None,
        lambda s: s.begin_turn(),
        lambda s: (s.begin_turn(), s.apply_lead_delta(LeadDelta(contact_name="Asha"))),
        lambda s: (s.begin_turn(), s.apply_lead_delta(WHOLESALE_DELTA)),
        lambda s: (s.begin_turn(), s.apply_lead_delta(WHOLESALE_DELTA), s.mark_handoff_ready()),
        lambda s: s.mark_declined(),
    ):
        state = ConversationState.new(SENDER)
        setup(state)
        assert state.mark_escalated("human requested") == QualificationState.ESCALATED
        assert state.escalation.status == EscalationStatus.PENDING
        assert state.escalation.reason == "human requested"
        assert state.escalation.requested_at_turn == state.turn_count
        assert state.is_terminal is True
        assert state.reevaluate_qualification() == QualificationState.ESCALATED


def test_escalation_pending_to_handed_off():
    state = ConversationState.new(SENDER)
    state.begin_turn()
    state.mark_escalated("complaint")
    state.begin_turn()
    assert state.mark_handed_off() == EscalationStatus.HANDED_OFF
    assert state.escalation.requested_at_turn == 1
    assert state.escalation.handed_off_at_turn == 2
    assert state.escalation.reason == "complaint"


def test_escalation_reason_is_bounded():
    state = ConversationState.new(SENDER)
    with pytest.raises(ValidationError):
        state.mark_escalated("x" * 201)


# ---------------------------------------------------------------------------
# 27. Flags serialization
# ---------------------------------------------------------------------------


def test_flags_defaults_and_serialization():
    flags = ConversationFlags()
    assert flags.model_dump() == {
        "injection_suspected": False,
        "injection_hits": 0,
        "anger_score": 0.0,
        "unanswered_asks": 0,
        "declines": 0,
        "repeated_question_count": 0,
        "off_topic_count": 0,
        "tool_failures_this_turn": 0,
        "grounding_violations": 0,
    }

    flags.injection_suspected = True
    flags.injection_hits = 2
    flags.anger_score = 0.75
    flags.unanswered_asks = 1
    flags.repeated_question_count = 3
    flags.off_topic_count = 4
    flags.grounding_violations = 1
    payload = json.loads(json.dumps(flags.model_dump(mode="json")))
    assert ConversationFlags.model_validate(payload) == flags


def test_flags_reject_negative_and_out_of_range_values():
    with pytest.raises(ValidationError):
        ConversationFlags(injection_hits=-1)
    with pytest.raises(ValidationError):
        ConversationFlags(anger_score=1.5)
    flags = ConversationFlags()
    with pytest.raises(ValidationError):
        flags.declines = -1
    with pytest.raises(ValidationError):
        ConversationFlags(unexpected=True)


# ---------------------------------------------------------------------------
# 28. ConversationStore get/create
# ---------------------------------------------------------------------------


def test_store_get_returns_none_for_unknown_sender():
    store = ConversationStore()
    assert store.get(SENDER) is None
    assert store.load(SENDER) is None
    assert SENDER not in store
    assert len(store) == 0


def test_store_get_or_create_makes_safe_default_once():
    store = ConversationStore()
    first = store.get_or_create(SENDER)
    assert first.sender_id == SENDER
    assert first.qualification == QualificationState.UNKNOWN
    assert first.lead.whatsapp_number == SENDER
    assert len(store) == 1

    second = store.get_or_create(SENDER)
    assert second == first
    assert len(store) == 1


def test_store_default_capacity_is_500():
    assert STORE_CAPACITY == 500
    assert ConversationStore().capacity == 500
    with pytest.raises(ValueError):
        ConversationStore(capacity=0)


# ---------------------------------------------------------------------------
# 29. ConversationStore update
# ---------------------------------------------------------------------------


def test_store_update_applies_mutator_and_persists():
    store = ConversationStore()

    def mutate(state: ConversationState) -> None:
        state.begin_turn()
        state.add_user_message("hello")
        state.set_intent(Intent.SMALLTALK, 0.6)

    returned = store.update(SENDER, mutate)
    assert returned.turn_count == 1
    assert returned.current_intent == Intent.SMALLTALK

    loaded = store.get(SENDER)
    assert loaded == returned
    assert loaded.history[0].content == "hello"


def test_store_hands_out_copies_not_references():
    store = ConversationStore()
    store.get_or_create(SENDER)
    working = store.get(SENDER)
    working.begin_turn()
    working.add_user_message("unsaved")
    # Not saved yet: the store is unchanged.
    assert store.get(SENDER).turn_count == 0
    assert store.get(SENDER).history == []
    store.save(working)
    assert store.get(SENDER).turn_count == 1


def test_store_update_leaves_store_unchanged_when_mutator_raises():
    store = ConversationStore()
    store.get_or_create(SENDER)

    def bad_mutate(state: ConversationState) -> None:
        state.begin_turn()
        state.add_user_message("")  # invalid → ValidationError

    with pytest.raises(ValidationError):
        store.update(SENDER, bad_mutate)
    assert store.get(SENDER).turn_count == 0


# ---------------------------------------------------------------------------
# 30. LRU eviction at capacity
# ---------------------------------------------------------------------------


def test_lru_eviction_at_capacity():
    store = ConversationStore(capacity=3)
    for sid in ("s1", "s2", "s3"):
        store.get_or_create(sid)
    assert store.sender_ids() == ["s1", "s2", "s3"]

    # Touch s1 so s2 becomes least recently used.
    store.get("s1")
    assert store.sender_ids() == ["s2", "s3", "s1"]

    store.get_or_create("s4")
    assert len(store) == 3
    assert "s2" not in store
    assert store.sender_ids() == ["s3", "s1", "s4"]

    # Saving an existing sender refreshes it without evicting anyone.
    store.save(store.get("s3"))
    assert len(store) == 3
    assert store.sender_ids() == ["s1", "s4", "s3"]


def test_lru_eviction_at_default_capacity():
    store = ConversationStore()
    for i in range(STORE_CAPACITY):
        store.get_or_create(f"sender-{i}")
    assert len(store) == STORE_CAPACITY
    store.get_or_create("one-more")
    assert len(store) == STORE_CAPACITY
    assert "sender-0" not in store
    assert "one-more" in store


# ---------------------------------------------------------------------------
# 31. Sender isolation
# ---------------------------------------------------------------------------


def test_sender_isolation():
    store = ConversationStore()
    a = "919876543210"
    b = "919876543211"

    store.update(a, lambda s: (s.begin_turn(), s.apply_lead_delta(WHOLESALE_DELTA)))
    store.update(b, lambda s: (s.begin_turn(), s.add_user_message("just browsing")))

    state_a = store.get(a)
    state_b = store.get(b)
    assert state_a.qualification == QualificationState.QUALIFIED
    assert state_b.qualification == QualificationState.UNKNOWN
    assert state_b.lead.business_name is None
    assert state_b.lead.whatsapp_number == b
    assert state_a.history == []
    assert state_b.history[0].content == "just browsing"

    store.delete(a)
    assert store.get(a) is None
    assert store.get(b) == state_b


# ---------------------------------------------------------------------------
# 32. State survives save/load
# ---------------------------------------------------------------------------


def test_state_survives_save_load_and_store_snapshot():
    store = ConversationStore(capacity=10)
    state = _qualified_state()
    state.set_intent(Intent.WHOLESALE_INQUIRY, 0.9)
    state.add_user_message("25kg per month please")
    state.remember_fact("wants_sample", "yes")
    state.flags.repeated_question_count = 2
    state.mark_handoff_ready()
    store.save(state)
    store.get_or_create("another")

    loaded = store.load(SENDER)
    assert loaded == state
    assert loaded is not state

    snapshot = store.snapshot()
    snapshot = json.loads(json.dumps(snapshot))  # plain JSON round-trip
    rebuilt = ConversationStore.from_snapshot(snapshot, capacity=10)
    assert len(rebuilt) == 2
    assert rebuilt.sender_ids() == store.sender_ids()
    assert rebuilt.get(SENDER) == state
    assert rebuilt.get(SENDER).qualification == QualificationState.HANDOFF_READY


def test_from_snapshot_rejects_mismatched_key():
    payload = ConversationState.new(SENDER).model_dump(mode="json")
    with pytest.raises(ValueError):
        ConversationStore.from_snapshot({"someone-else": payload})


def test_save_stores_validated_copy():
    store = ConversationStore()
    state = ConversationState.new(SENDER)
    store.save(state)
    state.begin_turn()
    assert store.get(SENDER).turn_count == 0


# ---------------------------------------------------------------------------
# 33. No external network activity
# ---------------------------------------------------------------------------


def test_agent_state_and_store_perform_no_network_access(monkeypatch):
    def _forbidden_socket(*args, **kwargs):
        raise AssertionError("agent state/store attempted to open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden_socket)
    monkeypatch.setattr(socket, "create_connection", _forbidden_socket)

    store = ConversationStore(capacity=5)
    state = store.update(
        SENDER,
        lambda s: (
            s.begin_turn(),
            s.add_user_message("hello"),
            s.set_intent(Intent.WHOLESALE_INQUIRY, 0.9),
            s.apply_lead_delta(WHOLESALE_DELTA),
            s.mark_handoff_ready(),
        ),
    )
    restored = ConversationState.model_validate(json.loads(json.dumps(state.model_dump(mode="json"))))
    assert restored == state
    assert ConversationStore.from_snapshot(store.snapshot(), capacity=5).get(SENDER) == state
