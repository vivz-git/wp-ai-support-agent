"""Deterministic escalation policy for the AI WhatsApp Support Agent.

``EscalationPolicy.evaluate(signals, state)`` turns this turn's guardrail
signals (``app.agent.guardrails``) plus the relevant ``ConversationState``
into one ``EscalationDecision``. Python owns every outcome here; the LLM
never decides to escalate, hand off, qualify, or refuse.

The policy is pure: it reads ``state`` and never mutates it. The future
orchestrator applies the decision (``mark_escalated``, ``mark_handoff_ready``,
suppressing a reply, ...) and updates flags with
``guardrails.apply_signals_to_flags``.

Call order contract: evaluate with the ``state`` *as it was before this
turn's signals were applied to its flags*. The policy adds the current
turn's contribution itself (e.g. prior ``injection_hits`` + this turn's hits),
so applying the flags first would double count.

Decision priority (lower number wins; every rule that fires is kept in
``reason_codes`` in this order):

     1  grounding violation        -> suppress      (unsafe output never ships)
     2  already escalated (sticky) -> escalate
     3  explicit human request     -> escalate
     4  high anger + complaint     -> escalate
     5  repeated unresolved ask    -> escalate
     6  repeated/aggressive injection -> escalate
     7  injection + secrets/internal data request -> refuse
     8  basic injection attempt    -> refuse
     9  qualified, complete lead   -> handoff_ready
    10  high anger, no complaint context -> clarify
    11  single repetition          -> clarify
    12  nothing fired              -> continue

Grounding outranks even a sticky escalation because an ungrounded reply
must not be sent regardless of who will handle the conversation next; the
orchestrator re-evaluates once it has a safe reply. Qualified handoff sits
below every escalation (a customer who needs a human is routed there even
if their lead profile is complete).

Security restrictions are never bypassed by qualification state (Milestone
2, Slice 14). Every injection rule (6, 7, 8) outranks the qualified-lead
rule (9), so a qualified, complete lead who sends an injection or a
secret/internal-data request gets the same ``refuse``/``escalate`` verdict
as any other sender; the ``lead_qualified`` code is still appended to
``reason_codes`` so the outcome stays explainable. Before Slice 14 the
qualified-lead rule sat at 7 and the two refusals at 8/9, so a qualified
lead's injection turn came back ``handoff_ready`` and reached the model
(pinned in the Slice 13 evaluation). Only those three ranks were swapped;
ranks 1-6 and 10-12 are unchanged, so an angry-but-not-complaining or
once-repeating qualified lead is still ``handoff_ready`` (9 beats 10/11).

Determinism: the policy is pure, synchronous Python over ``signals`` and
``state``; no model output reaches it (``GuardrailSignals`` carries scores,
counts and codes only, and the grounding verdict is computed by the
validator, never read from the reply). Identical inputs always give an
identical ``EscalationDecision``: candidates are sorted by ``(priority,
code)`` and ``reason_codes`` is that sorted order.
"""

from enum import Enum
from typing import List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.agent.guardrails import GuardrailSignals
from app.agent.lead import QualificationState
from app.agent.state import ConversationState, EscalationStatus, Intent

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

ANGER_ESCALATION_THRESHOLD = 0.6
# Total consecutive repeats (prior count + this turn) at which we escalate;
# a single repeat only asks for clarification.
REPEAT_ESCALATION_THRESHOLD = 2
# Cumulative matched injection patterns (prior hits + this turn) at which an
# attempt counts as repeated/aggressive.
INJECTION_ESCALATION_HITS = 3

_COMPLAINT_INTENTS = frozenset({Intent.COMPLAINT, Intent.ORDER_STATUS})


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------


class EscalationAction(str, Enum):
    CONTINUE = "continue"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    SUPPRESS = "suppress"  # do not send the generated reply as-is
    ESCALATE = "escalate"
    HANDOFF_READY = "handoff_ready"


class UserMessageInstruction(str, Enum):
    CONTINUE_NORMAL_RESPONSE = "continue_normal_response"
    ASK_FOR_CLARIFICATION = "ask_for_clarification"
    PROVIDE_SAFE_REFUSAL = "provide_safe_refusal"
    SUPPRESS_UNGROUNDED_CLAIM = "suppress_ungrounded_claim"
    OFFER_HUMAN_HANDOFF = "offer_human_handoff"


class EscalationDecision(BaseModel):
    """The policy's verdict for one turn. Codes only; never customer text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action: EscalationAction
    reason_codes: List[str] = Field(default_factory=list)
    user_message_instruction: UserMessageInstruction
    priority: int = Field(..., ge=1)

    @property
    def blocks_reply(self) -> bool:
        return self.action == EscalationAction.SUPPRESS

    @property
    def escalates(self) -> bool:
        return self.action == EscalationAction.ESCALATE


# (priority, action, reason_code, instruction)
_Candidate = Tuple[int, EscalationAction, str, UserMessageInstruction]

_CONTINUE_PRIORITY = 12


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class EscalationPolicy:
    """Deterministic rule set over guardrail signals and conversation state."""

    def __init__(
        self,
        anger_threshold: float = ANGER_ESCALATION_THRESHOLD,
        repeat_threshold: int = REPEAT_ESCALATION_THRESHOLD,
        injection_escalation_hits: int = INJECTION_ESCALATION_HITS,
    ):
        if not 0.0 < anger_threshold <= 1.0:
            raise ValueError("anger_threshold must be in (0, 1]")
        if repeat_threshold < 1 or injection_escalation_hits < 1:
            raise ValueError("thresholds must be at least 1")
        self._anger_threshold = anger_threshold
        self._repeat_threshold = repeat_threshold
        self._injection_escalation_hits = injection_escalation_hits

    def evaluate(self, signals: GuardrailSignals, state: ConversationState) -> EscalationDecision:
        candidates: List[_Candidate] = []
        candidates.extend(self._grounding_rule(signals))
        candidates.extend(self._sticky_escalation_rule(state))
        candidates.extend(self._human_request_rule(signals, state))
        candidates.extend(self._anger_rules(signals, state))
        candidates.extend(self._repetition_rules(signals, state))
        candidates.extend(self._injection_rules(signals, state))
        candidates.extend(self._qualified_lead_rule(state))

        if not candidates:
            return EscalationDecision(
                action=EscalationAction.CONTINUE,
                reason_codes=["no_rule_fired"],
                user_message_instruction=UserMessageInstruction.CONTINUE_NORMAL_RESPONSE,
                priority=_CONTINUE_PRIORITY,
            )

        candidates.sort(key=lambda c: (c[0], c[2]))  # priority, then code: fully deterministic
        priority, action, _, instruction = candidates[0]
        return EscalationDecision(
            action=action,
            reason_codes=[code for _, _, code, _ in candidates],
            user_message_instruction=instruction,
            priority=priority,
        )

    # -- Rules ----------------------------------------------------------------

    @staticmethod
    def _grounding_rule(signals: GuardrailSignals) -> List[_Candidate]:
        grounding = signals.grounding
        if grounding is None or grounding.grounded:
            return []
        return [(1, EscalationAction.SUPPRESS, "grounding_violation", UserMessageInstruction.SUPPRESS_UNGROUNDED_CLAIM)]

    @staticmethod
    def _sticky_escalation_rule(state: ConversationState) -> List[_Candidate]:
        escalated = (
            state.qualification == QualificationState.ESCALATED
            or state.escalation.status in (EscalationStatus.PENDING, EscalationStatus.HANDED_OFF)
        )
        if not escalated:
            return []
        return [(2, EscalationAction.ESCALATE, "already_escalated", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]

    @staticmethod
    def _human_request_rule(signals: GuardrailSignals, state: ConversationState) -> List[_Candidate]:
        requested = signals.human_request is not None and signals.human_request.requested
        if not requested and state.current_intent != Intent.HUMAN_REQUEST:
            return []
        return [(3, EscalationAction.ESCALATE, "human_requested", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]

    def _anger_rules(self, signals: GuardrailSignals, state: ConversationState) -> List[_Candidate]:
        anger = signals.anger
        if anger is None or anger.score < self._anger_threshold:
            return []
        if self._complaint_context(signals, state):
            return [(4, EscalationAction.ESCALATE, "high_anger_complaint", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]
        return [(10, EscalationAction.CLARIFY, "high_anger_no_complaint_context", UserMessageInstruction.ASK_FOR_CLARIFICATION)]

    @staticmethod
    def _complaint_context(signals: GuardrailSignals, state: ConversationState) -> bool:
        """Anger only escalates when something has actually gone wrong."""
        if signals.anger is not None and signals.anger.complaint_language:
            return True
        if state.current_intent in _COMPLAINT_INTENTS:
            return True
        if any(record.intent in _COMPLAINT_INTENTS for record in state.intent_history):
            return True
        flags = state.flags
        return flags.repeated_question_count > 0 or flags.unanswered_asks > 0 or flags.tool_failures_this_turn > 0

    def _repetition_rules(self, signals: GuardrailSignals, state: ConversationState) -> List[_Candidate]:
        repetition = signals.repetition
        if repetition is None or not repetition.repeated:
            return []
        total_repeats = state.flags.repeated_question_count + 1
        if total_repeats >= self._repeat_threshold:
            return [(5, EscalationAction.ESCALATE, "repeated_unresolved", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]
        return [(11, EscalationAction.CLARIFY, "repeated_question", UserMessageInstruction.ASK_FOR_CLARIFICATION)]

    def _injection_rules(self, signals: GuardrailSignals, state: ConversationState) -> List[_Candidate]:
        injection = signals.injection
        if injection is None or not injection.suspected:
            return []
        prior_hits = state.flags.injection_hits
        cumulative_hits = prior_hits + len(injection.hits)
        repeated_attempt = prior_hits > 0 and state.flags.injection_suspected
        aggressive = cumulative_hits >= self._injection_escalation_hits or (
            injection.secrets_requested and repeated_attempt
        )
        candidates: List[_Candidate] = []
        if aggressive:
            candidates.append(
                (6, EscalationAction.ESCALATE, "injection_repeated", UserMessageInstruction.OFFER_HUMAN_HANDOFF)
            )
        # Slice 14: both refusals (7, 8) outrank the qualified-lead rule (9).
        if injection.internal_data_requested:
            code = "injection_secrets_requested" if injection.secrets_requested else "injection_internal_data_requested"
            candidates.append((7, EscalationAction.REFUSE, code, UserMessageInstruction.PROVIDE_SAFE_REFUSAL))
        else:
            candidates.append((8, EscalationAction.REFUSE, "injection_attempt", UserMessageInstruction.PROVIDE_SAFE_REFUSAL))
        return candidates

    @staticmethod
    def _qualified_lead_rule(state: ConversationState) -> List[_Candidate]:
        """Deterministic qualification only: the model never declares this.

        Rank 9 (Slice 14): below every escalation and every injection
        refusal, so qualification state can never bypass a security rule.
        """
        if state.qualification == QualificationState.HANDOFF_READY:
            return [(9, EscalationAction.HANDOFF_READY, "handoff_ready", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]
        if state.qualification != QualificationState.QUALIFIED or not state.lead.is_complete():
            return []
        return [(9, EscalationAction.HANDOFF_READY, "lead_qualified", UserMessageInstruction.OFFER_HUMAN_HANDOFF)]


__all__ = [
    "ANGER_ESCALATION_THRESHOLD",
    "EscalationAction",
    "EscalationDecision",
    "EscalationPolicy",
    "INJECTION_ESCALATION_HITS",
    "REPEAT_ESCALATION_THRESHOLD",
    "UserMessageInstruction",
]
