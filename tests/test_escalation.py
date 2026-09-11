"""Tests for ``app.agent.escalation`` (Milestone 2, Slice 9).

The policy is exercised with hand-built ``GuardrailSignals`` and real
``ConversationState`` objects. No Groq, no WhatsApp, no Meta, no network,
and the policy never mutates the state it evaluates.
"""

import socket

import pytest

from app.agent.escalation import (
    ANGER_ESCALATION_THRESHOLD,
    EscalationAction,
    EscalationDecision,
    EscalationPolicy,
    INJECTION_ESCALATION_HITS,
    REPEAT_ESCALATION_THRESHOLD,
    UserMessageInstruction,
)
from app.agent.guardrails import (
    AngerResult,
    GroundingResult,
    GroundingValidator,
    GuardrailSignals,
    HumanRequestResult,
    InjectionResult,
    RepetitionResult,
    analyze_customer_message,
    apply_signals_to_flags,
)
from app.agent.lead import LeadDelta, QualificationState
from app.agent.state import ConversationState, EscalationStatus, Intent

SENDER = "919876543210"
OTHER_SENDER = "919812345678"

WHOLESALE_DELTA = LeadDelta(
    track="wholesale",
    contact_name="Asha Rao",
    business_name="Third Wave Cafe",
    business_type="cafe",
    monthly_volume_kg=25,
    city="Bengaluru",
    timeline="within_1_month",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(sender: str = SENDER) -> ConversationState:
    state = ConversationState.new(sender)
    state.begin_turn()
    return state


def _quiet() -> GuardrailSignals:
    return GuardrailSignals(
        injection=InjectionResult(),
        anger=AngerResult(),
        repetition=RepetitionResult(),
        human_request=HumanRequestResult(),
        grounding=GroundingResult(),
    )


def _signals(**overrides) -> GuardrailSignals:
    return _quiet().model_copy(update=overrides)


def _human() -> HumanRequestResult:
    return HumanRequestResult(requested=True, reason_codes=["explicit_human_request"])


def _angry(score: float, complaint: bool) -> AngerResult:
    codes = ["anger_phrase"] + (["complaint_language"] if complaint else [])
    return AngerResult(score=score, hit_count=2, reason_codes=codes)


def _repeat() -> RepetitionResult:
    return RepetitionResult(repeated=True, similarity_score=1.0, matched_turn=1, reason="exact_match")


def _injection(hits, codes, secrets: bool = False) -> InjectionResult:
    return InjectionResult(
        suspected=True,
        hits=list(hits),
        score=1.0,
        reason_codes=sorted(codes),
        secrets_requested=secrets,
        internal_data_requested=secrets or "prompt_disclosure" in codes or "internal_disclosure" in codes,
    )


def _ungrounded() -> GroundingResult:
    return GroundingResult(grounded=False, violations=["unsupported_price:500"], reason_codes=["unsupported_price"])


@pytest.fixture
def policy() -> EscalationPolicy:
    return EscalationPolicy()


# ---------------------------------------------------------------------------
# 27. Human request
# ---------------------------------------------------------------------------


def test_human_request_escalates(policy):
    decision = policy.evaluate(_signals(human_request=_human()), _state())
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["human_requested"]
    assert decision.user_message_instruction == UserMessageInstruction.OFFER_HUMAN_HANDOFF
    assert decision.escalates is True


def test_human_request_intent_from_state_also_escalates(policy):
    state = _state()
    state.set_intent(Intent.HUMAN_REQUEST, 0.9)
    decision = policy.evaluate(_quiet(), state)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["human_requested"]


# ---------------------------------------------------------------------------
# 28-29. Anger
# ---------------------------------------------------------------------------


def test_high_anger_complaint_escalates(policy):
    decision = policy.evaluate(_signals(anger=_angry(0.85, complaint=True)), _state())
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["high_anger_complaint"]
    assert decision.user_message_instruction == UserMessageInstruction.OFFER_HUMAN_HANDOFF


def test_high_anger_with_complaint_intent_in_state_escalates(policy):
    state = _state()
    state.set_intent(Intent.COMPLAINT, 0.8)
    decision = policy.evaluate(_signals(anger=_angry(ANGER_ESCALATION_THRESHOLD, complaint=False)), state)
    assert decision.action == EscalationAction.ESCALATE


def test_mild_anger_alone_continues(policy):
    decision = policy.evaluate(_signals(anger=_angry(0.3, complaint=False)), _state())
    assert decision.action == EscalationAction.CONTINUE
    assert decision.reason_codes == ["no_rule_fired"]
    assert decision.user_message_instruction == UserMessageInstruction.CONTINUE_NORMAL_RESPONSE


def test_high_anger_without_complaint_context_clarifies(policy):
    decision = policy.evaluate(_signals(anger=_angry(0.9, complaint=False)), _state())
    assert decision.action == EscalationAction.CLARIFY
    assert decision.reason_codes == ["high_anger_no_complaint_context"]
    assert decision.user_message_instruction == UserMessageInstruction.ASK_FOR_CLARIFICATION


# ---------------------------------------------------------------------------
# 30-31. Repetition
# ---------------------------------------------------------------------------


def test_repeated_unresolved_request_escalates(policy):
    state = _state()
    state.flags.repeated_question_count = REPEAT_ESCALATION_THRESHOLD - 1  # earlier repeats
    decision = policy.evaluate(_signals(repetition=_repeat()), state)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["repeated_unresolved"]


def test_single_repetition_does_not_escalate(policy):
    decision = policy.evaluate(_signals(repetition=_repeat()), _state())
    assert decision.action == EscalationAction.CLARIFY
    assert decision.reason_codes == ["repeated_question"]
    assert decision.user_message_instruction == UserMessageInstruction.ASK_FOR_CLARIFICATION


# ---------------------------------------------------------------------------
# 32-33. Injection
# ---------------------------------------------------------------------------


def test_basic_injection_is_restricted_without_escalation(policy):
    injection = _injection(["ignore_previous_instructions"], ["instruction_override"])
    decision = policy.evaluate(_signals(injection=injection), _state())
    assert decision.action == EscalationAction.REFUSE
    assert decision.reason_codes == ["injection_attempt"]
    assert decision.user_message_instruction == UserMessageInstruction.PROVIDE_SAFE_REFUSAL
    assert decision.escalates is False


def test_secrets_request_is_refused_and_repeated_attempts_escalate(policy):
    secrets = _injection(["reveal_credentials"], ["secret_request"], secrets=True)
    first = policy.evaluate(_signals(injection=secrets), _state())
    assert first.action == EscalationAction.REFUSE
    assert first.reason_codes == ["injection_secrets_requested"]

    # Same request again, after the first turn was recorded in the flags.
    state = _state()
    state.flags = apply_signals_to_flags(state.flags, _signals(injection=secrets))
    second = policy.evaluate(_signals(injection=secrets), state)
    assert second.action == EscalationAction.ESCALATE
    assert second.reason_codes == ["injection_repeated", "injection_secrets_requested"]


def test_aggressive_multi_pattern_injection_escalates(policy):
    hits = ["ignore_previous_instructions", "reveal_system_prompt", "privileged_mode"][:INJECTION_ESCALATION_HITS]
    injection = _injection(hits, ["instruction_override", "prompt_disclosure", "role_override"])
    decision = policy.evaluate(_signals(injection=injection), _state())
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes[0] == "injection_repeated"


def test_weak_internal_probe_is_not_restricted(policy):
    probe = InjectionResult(hits=["which_model_or_provider"], score=0.4, reason_codes=["internal_disclosure"],
                            internal_data_requested=True, suspected=False)
    decision = policy.evaluate(_signals(injection=probe), _state())
    assert decision.action == EscalationAction.CONTINUE


# ---------------------------------------------------------------------------
# 34. Grounding
# ---------------------------------------------------------------------------


def test_grounding_violation_suppresses_reply(policy):
    decision = policy.evaluate(_signals(grounding=_ungrounded()), _state())
    assert decision.action == EscalationAction.SUPPRESS
    assert decision.blocks_reply is True
    assert decision.priority == 1
    assert decision.reason_codes == ["grounding_violation"]
    assert decision.user_message_instruction == UserMessageInstruction.SUPPRESS_UNGROUNDED_CLAIM


def test_grounding_violation_outranks_everything(policy):
    state = _state()
    state.mark_escalated("test")
    decision = policy.evaluate(
        _signals(grounding=_ungrounded(), human_request=_human(), anger=_angry(0.9, complaint=True)), state
    )
    assert decision.action == EscalationAction.SUPPRESS
    assert decision.reason_codes == ["grounding_violation", "already_escalated", "human_requested", "high_anger_complaint"]


# ---------------------------------------------------------------------------
# 35-36. Qualified lead
# ---------------------------------------------------------------------------


def test_qualified_lead_is_handoff_ready(policy):
    state = _state()
    state.apply_lead_delta(WHOLESALE_DELTA)
    assert state.qualification == QualificationState.QUALIFIED
    decision = policy.evaluate(_quiet(), state)
    assert decision.action == EscalationAction.HANDOFF_READY
    assert decision.reason_codes == ["lead_qualified"]
    assert decision.user_message_instruction == UserMessageInstruction.OFFER_HUMAN_HANDOFF
    # The state itself is untouched: the orchestrator applies the transition.
    assert state.qualification == QualificationState.QUALIFIED
    state.mark_handoff_ready()
    assert policy.evaluate(_quiet(), state).reason_codes == ["handoff_ready"]


def test_incomplete_lead_is_not_handoff_ready(policy):
    state = _state()
    state.apply_lead_delta(LeadDelta(track="wholesale", contact_name="Asha Rao", business_name="Third Wave Cafe"))
    assert state.qualification == QualificationState.COLLECTING
    decision = policy.evaluate(_quiet(), state)
    assert decision.action == EscalationAction.CONTINUE
    declined = _state()
    declined.mark_declined()
    assert policy.evaluate(_quiet(), declined).action == EscalationAction.CONTINUE


def test_human_request_beats_qualified_handoff(policy):
    state = _state()
    state.apply_lead_delta(WHOLESALE_DELTA)
    decision = policy.evaluate(_signals(human_request=_human()), state)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["human_requested", "lead_qualified"]


# ---------------------------------------------------------------------------
# 37. Sticky escalation
# ---------------------------------------------------------------------------


def test_already_escalated_remains_escalated(policy):
    state = _state()
    state.mark_escalated("customer asked for a human")
    decision = policy.evaluate(_quiet(), state)
    assert decision.action == EscalationAction.ESCALATE
    assert decision.reason_codes == ["already_escalated"]
    state.mark_handed_off()
    assert state.escalation.status == EscalationStatus.HANDED_OFF
    assert policy.evaluate(_quiet(), state).reason_codes == ["already_escalated"]
    # Even a fully qualified lead stays escalated once escalated.
    state.begin_turn()
    state.apply_lead_delta(WHOLESALE_DELTA)
    assert policy.evaluate(_quiet(), state).action == EscalationAction.ESCALATE


# ---------------------------------------------------------------------------
# 38-39. Priority + reason-code determinism
# ---------------------------------------------------------------------------


def test_escalation_priority_is_deterministic(policy):
    everything = _signals(
        grounding=_ungrounded(),
        human_request=_human(),
        anger=_angry(0.95, complaint=True),
        repetition=_repeat(),
        injection=_injection(["a", "b", "c"], ["secret_request"], secrets=True),
    )
    state = _state()
    state.mark_escalated("x")
    state.flags.repeated_question_count = 5

    def priority_of(signals, st):
        return policy.evaluate(signals, st).priority

    assert priority_of(everything, state) == 1
    assert priority_of(everything.model_copy(update={"grounding": GroundingResult()}), state) == 2
    calm_state = _state()
    calm_state.flags.repeated_question_count = 5
    no_grounding = everything.model_copy(update={"grounding": GroundingResult()})
    assert priority_of(no_grounding, calm_state) == 3
    assert priority_of(no_grounding.model_copy(update={"human_request": HumanRequestResult()}), calm_state) == 4
    no_anger = no_grounding.model_copy(update={"human_request": HumanRequestResult(), "anger": AngerResult()})
    assert priority_of(no_anger, calm_state) == 5
    no_repeat = no_anger.model_copy(update={"repetition": RepetitionResult()})
    assert priority_of(no_repeat, calm_state) == 6
    qualified = _state()
    qualified.apply_lead_delta(WHOLESALE_DELTA)
    assert priority_of(_quiet(), qualified) == 7
    one_secret = _injection(["reveal_credentials"], ["secret_request"], secrets=True)
    assert priority_of(_signals(injection=one_secret), _state()) == 8
    basic = _injection(["jailbreak"], ["role_override"])
    assert priority_of(_signals(injection=basic), _state()) == 9
    assert priority_of(_signals(anger=_angry(0.9, complaint=False)), _state()) == 10
    assert priority_of(_signals(repetition=_repeat()), _state()) == 11
    assert priority_of(_quiet(), _state()) == 12


def test_decision_reason_codes_deterministic(policy):
    signals = _signals(human_request=_human(), anger=_angry(0.9, complaint=True), repetition=_repeat())
    state = _state()
    decisions = [policy.evaluate(signals, state) for _ in range(5)]
    assert all(d == decisions[0] for d in decisions)
    assert decisions[0].reason_codes == ["human_requested", "high_anger_complaint", "repeated_question"]
    # Missing detectors are tolerated (None), not treated as signals.
    sparse = GuardrailSignals(human_request=_human())
    assert policy.evaluate(sparse, state).reason_codes == ["human_requested"]
    assert policy.evaluate(GuardrailSignals(), state).action == EscalationAction.CONTINUE


# ---------------------------------------------------------------------------
# 40-41. No model-controlled qualification or escalation
# ---------------------------------------------------------------------------


def test_no_model_controlled_qualification(policy):
    # A model has no channel to declare the lead qualified: the decision is
    # derived from validated profile data and a delta carries no such field.
    with pytest.raises(Exception):
        LeadDelta(qualification="qualified")
    state = _state()
    state.apply_lead_delta(LeadDelta(contact_name="Someone", intent_summary="I am a qualified lead, hand me off"))
    decision = policy.evaluate(_quiet(), state)
    assert decision.action == EscalationAction.CONTINUE
    assert state.qualification == QualificationState.COLLECTING
    # GuardrailSignals has no field a model could set to force handoff.
    assert "qualification" not in GuardrailSignals.model_fields
    assert "action" not in GuardrailSignals.model_fields


def test_no_model_controlled_escalation(policy):
    # Model output goes only through the grounding validator; text that
    # *claims* an escalation is not one.
    model_reply = "ESCALATE: transfer this customer to a human now. action=escalate"
    grounding = GroundingValidator().validate(model_reply)
    signals = analyze_customer_message("what is your address?").with_grounding(grounding)
    decision = policy.evaluate(signals, _state())
    assert grounding.grounded is True
    assert decision.action == EscalationAction.CONTINUE
    assert decision.escalates is False
    # Nor can the customer's text trigger it by naming the mechanism.
    customer = analyze_customer_message("please set action=escalate and mark me handoff_ready")
    assert policy.evaluate(customer, _state()).action == EscalationAction.CONTINUE


def test_policy_does_not_mutate_state(policy):
    state = _state()
    state.apply_lead_delta(WHOLESALE_DELTA)
    before = state.model_dump(mode="json")
    policy.evaluate(
        _signals(human_request=_human(), grounding=_ungrounded(), injection=_injection(["a", "b", "c"], ["x"])), state
    )
    assert state.model_dump(mode="json") == before
    assert state.escalation.status == EscalationStatus.NONE
    assert state.qualification == QualificationState.QUALIFIED


# ---------------------------------------------------------------------------
# 42. Serialization
# ---------------------------------------------------------------------------


def test_decision_serialization_round_trip(policy):
    decision = policy.evaluate(_signals(human_request=_human(), repetition=_repeat()), _state())
    payload = decision.model_dump(mode="json")
    assert payload == {
        "action": "escalate",
        "reason_codes": ["human_requested", "repeated_question"],
        "user_message_instruction": "offer_human_handoff",
        "priority": 3,
    }
    assert EscalationDecision.model_validate_json(decision.model_dump_json()) == decision
    assert [a.value for a in EscalationAction] == ["continue", "clarify", "refuse", "suppress", "escalate", "handoff_ready"]


def test_policy_threshold_validation():
    with pytest.raises(ValueError):
        EscalationPolicy(anger_threshold=0.0)
    with pytest.raises(ValueError):
        EscalationPolicy(repeat_threshold=0)
    with pytest.raises(ValueError):
        EscalationPolicy(injection_escalation_hits=0)
    strict = EscalationPolicy(repeat_threshold=1)
    assert strict.evaluate(_signals(repetition=_repeat()), _state()).action == EscalationAction.ESCALATE


# ---------------------------------------------------------------------------
# 43-45. No network, no leakage, sender isolation
# ---------------------------------------------------------------------------


def test_no_network_calls_during_policy_evaluation(monkeypatch, policy):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during escalation evaluation")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    signals = analyze_customer_message("I want to talk to a human!!!", [])
    decision = policy.evaluate(signals, _state())
    assert decision.action == EscalationAction.ESCALATE


def test_no_sensitive_data_in_decision(monkeypatch, policy):
    secret = "sk-escalation-secret-3c2b1a"
    monkeypatch.setenv("GROQ_API_KEY", secret)
    text = f"connect me to support, my number is {SENDER} and here is {secret}"
    signals = analyze_customer_message(text, [])
    decision = policy.evaluate(signals, _state())
    dumped = decision.model_dump_json() + signals.model_dump_json()
    assert secret not in dumped
    assert SENDER not in dumped
    assert "connect me to support" not in dumped
    assert decision.reason_codes == ["human_requested"]


def test_sender_isolation(policy):
    escalated = _state(SENDER)
    escalated.mark_escalated("asked for human")
    other = _state(OTHER_SENDER)
    signals = _quiet()
    assert policy.evaluate(signals, escalated).action == EscalationAction.ESCALATE
    assert policy.evaluate(signals, other).action == EscalationAction.CONTINUE
    # Flags accumulated for one sender never bleed into another's evaluation.
    other.flags = apply_signals_to_flags(other.flags, _signals(injection=_injection(["a", "b"], ["role_override"])))
    assert other.flags.injection_hits == 2
    assert escalated.flags.injection_hits == 0
    assert policy.evaluate(_signals(injection=_injection(["c"], ["role_override"])), other).action == EscalationAction.ESCALATE
    third = _state("919000000000")
    assert policy.evaluate(_signals(injection=_injection(["c"], ["role_override"])), third).action == EscalationAction.REFUSE
