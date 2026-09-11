"""Agent domain model for the AI WhatsApp Support Agent.

Design constraints (Milestone 2, Slice 4):
- Pure state + lead-qualification domain model. No orchestrator, no
  prompts, no guardrails, no escalation rules, no handoff sink.
- No network, no database, no LLM calls.
- Not imported by ``app.main``: the running Milestone 1 webhook path is
  unchanged. Wiring is future-slice work.
"""

from app.agent.lead import (
    BudgetBand,
    BusinessType,
    LeadDelta,
    LeadProfile,
    LeadSource,
    LeadTrack,
    QualificationState,
    Timeline,
    evaluate_qualification,
    merge_lead_delta,
)
from app.agent.state import (
    ConversationFlags,
    ConversationState,
    EscalationState,
    EscalationStatus,
    HistoryMessage,
    Intent,
    IntentRecord,
    InvalidTransitionError,
    ToolInvocation,
)
from app.agent.store import ConversationStore

__all__ = [
    "BudgetBand",
    "BusinessType",
    "ConversationFlags",
    "ConversationState",
    "ConversationStore",
    "EscalationState",
    "EscalationStatus",
    "HistoryMessage",
    "Intent",
    "IntentRecord",
    "InvalidTransitionError",
    "LeadDelta",
    "LeadProfile",
    "LeadSource",
    "LeadTrack",
    "QualificationState",
    "Timeline",
    "ToolInvocation",
    "evaluate_qualification",
    "merge_lead_delta",
]
