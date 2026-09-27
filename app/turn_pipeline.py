"""Shared turn pipeline: idempotency ledger -> ``AgentOrchestrator`` -> draft queue.

Both the real WhatsApp webhook (``app.main.receive_webhook``) and the
``/simulate`` demo page (``app.simulate``) call ``process_incoming_message``
so a simulated patient message runs through exactly the same code as a real
one — the only difference is where the ``sender``/``text``/``message_id``
came from (a parsed Meta payload vs. a typed demo message).

This module never sends anything itself: the reply is always queued as a
pending draft. Delivery happens only from ``/staff`` (or, for a simulated
``SIM-`` sender, by being shown in the simulator instead of a real send —
see ``app.simulate``).
"""

import logging
from typing import Any, Dict

from app.agent.escalation import EscalationAction
from app.agent.orchestrator import AgentOrchestrator
from app.approval import DraftQueue
from app.config import mask_phone_number
from app.memory import InMemoryConversationMemory

logger = logging.getLogger(__name__)


async def process_incoming_message(
    sender: str,
    text: str,
    message_id: str,
    memory: InMemoryConversationMemory,
    orchestrator: AgentOrchestrator,
    drafts: DraftQueue,
) -> Dict[str, Any]:
    """Run one customer/patient turn end to end. Never sends; never raises.

    Returns the same status dict shape callers hand back as their response:
    ``duplicate_ignored``, ``ignored`` (empty text), ``agent_error``,
    ``queue_error``, or ``pending_approval`` with the new draft's id and
    urgency.
    """
    masked_sender = mask_phone_number(sender)

    # Idempotency: a retried delivery (e.g. Meta re-sending after a non-200
    # response) must not re-run the agent and queue a second duplicate draft.
    if memory.has_processed(message_id):
        logger.info(
            "Ignored duplicate delivery for message_id=%s (sender %s)",
            message_id,
            masked_sender,
        )
        return {"status": "duplicate_ignored", "message_id": message_id}

    memory.mark_processed(message_id)

    if not text.strip():
        logger.info("Ignored empty text message from %s (id: %s)", masked_sender, message_id)
        return {"status": "ignored", "reason": "empty_text", "message_id": message_id}

    logger.info("Processing incoming message from %s (id: %s)", masked_sender, message_id)

    try:
        turn_result = await orchestrator.handle_turn(sender_id=sender, text=text, message_id=message_id)
    except Exception as exc:
        logger.error("Agent turn failed for %s: %s", masked_sender, type(exc).__name__)
        logger.debug("Agent turn failure detail", exc_info=exc)
        return {"status": "agent_error", "message_id": message_id}

    # Human approval layer: the reply is never sent from here. Staff approve
    # (optionally edit) it on /staff, which is the only path to wa_client.send_text
    # (or, for a simulated patient, to being shown in /simulate instead).
    is_urgent = turn_result.diagnostics.escalation_action == EscalationAction.ESCALATE.value
    try:
        draft = drafts.add(
            sender=sender,
            patient_message=text,
            draft_text=turn_result.reply_text,
            is_urgent=is_urgent,
        )
    except Exception as exc:
        logger.error("Failed to queue reply draft for %s: %s", masked_sender, type(exc).__name__)
        logger.debug("Draft queue failure detail", exc_info=exc)
        return {"status": "queue_error", "message_id": message_id}

    logger.info(
        "Queued reply draft %d for %s (urgent=%s); awaiting staff approval",
        draft.id,
        masked_sender,
        is_urgent,
    )
    return {
        "status": "pending_approval",
        "message_id": message_id,
        "draft_id": draft.id,
        "is_urgent": is_urgent,
    }
