"""Agent domain model for the AI WhatsApp Support Agent: conversation state,
patient lead profile and qualification. No network, no database, no LLM calls.
"""

from app.agent.lead import (
    LeadDelta,
    LeadProfile,
    LeadSource,
    QualificationState,
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
    "QualificationState",
    "ToolInvocation",
    "evaluate_qualification",
    "merge_lead_delta",
]
