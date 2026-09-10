import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class IncomingTextMessage(BaseModel):
    """Normalized representation of an incoming WhatsApp text message."""
    sender: str = Field(..., description="Sender WhatsApp phone number")
    text: str = Field(..., description="Text content of the message")
    message_id: str = Field(..., description="Meta WhatsApp message ID (wamid)")
    timestamp: Optional[str] = Field(None, description="Message timestamp from Meta")


class WebhookParseResult(BaseModel):
    """Result of parsing an incoming Meta webhook event."""
    is_valid_structure: bool = Field(..., description="True if payload matches Meta webhook structure")
    event_type: str = Field(..., description="Classification of the webhook event")
    message: Optional[IncomingTextMessage] = Field(None, description="Parsed text message if applicable")
    reason: Optional[str] = Field(None, description="Human-readable reason for ignored/unsupported events")


def parse_incoming_webhook(payload: Any) -> WebhookParseResult:
    """Safely inspect and parse a Meta WhatsApp Cloud API webhook payload.
    
    Args:
        payload: Decoded JSON payload from Meta webhook POST request.
        
    Returns:
        WebhookParseResult detailing event type and extracted text message if valid.
    """
    if not isinstance(payload, dict):
        return WebhookParseResult(
            is_valid_structure=False,
            event_type="malformed",
            reason="Payload must be a JSON object",
        )

    if "object" not in payload or "entry" not in payload:
        return WebhookParseResult(
            is_valid_structure=False,
            event_type="malformed",
            reason="Payload missing required 'object' or 'entry' fields",
        )

    if payload.get("object") != "whatsapp_business_account":
        return WebhookParseResult(
            is_valid_structure=True,
            event_type="unsupported_object",
            reason=f"Ignored non-whatsapp object: {payload.get('object')}",
        )

    entries = payload.get("entry")
    if not isinstance(entries, list) or len(entries) == 0:
        return WebhookParseResult(
            is_valid_structure=True,
            event_type="empty_entry",
            reason="Payload contains empty entry list",
        )

    # Process changes in entry
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes")
        if not isinstance(changes, list) or len(changes) == 0:
            continue

        for change in changes:
            if not isinstance(change, dict):
                continue
            if change.get("field") != "messages":
                continue

            value = change.get("value")
            if not isinstance(value, dict):
                continue

            # Check if this is a message delivery status update (sent, delivered, read)
            statuses = value.get("statuses")
            messages = value.get("messages")

            if statuses and not messages:
                status_name = statuses[0].get("status") if isinstance(statuses, list) and len(statuses) > 0 and isinstance(statuses[0], dict) else "unknown"
                return WebhookParseResult(
                    is_valid_structure=True,
                    event_type="status_update",
                    reason=f"Ignored status notification ({status_name})",
                )

            if not messages or not isinstance(messages, list) or len(messages) == 0:
                return WebhookParseResult(
                    is_valid_structure=True,
                    event_type="no_messages",
                    reason="Webhook change value has no messages",
                )

            raw_msg = messages[0]
            if not isinstance(raw_msg, dict):
                continue

            msg_type = raw_msg.get("type")
            if msg_type != "text":
                return WebhookParseResult(
                    is_valid_structure=True,
                    event_type=f"unsupported_media_{msg_type}",
                    reason=f"Ignored non-text message type: {msg_type}",
                )

            sender = raw_msg.get("from")
            message_id = raw_msg.get("id") or "unknown_id"
            timestamp = str(raw_msg.get("timestamp")) if raw_msg.get("timestamp") else None
            text_container = raw_msg.get("text")

            body: Optional[str] = None
            if isinstance(text_container, dict):
                body = text_container.get("body")

            if not sender or not body:
                return WebhookParseResult(
                    is_valid_structure=True,
                    event_type="empty_message",
                    reason="Message is missing sender phone or body text",
                )

            return WebhookParseResult(
                is_valid_structure=True,
                event_type="text_message",
                message=IncomingTextMessage(
                    sender=str(sender),
                    text=str(body),
                    message_id=str(message_id),
                    timestamp=timestamp,
                ),
            )

    return WebhookParseResult(
        is_valid_structure=True,
        event_type="no_actionable_events",
        reason="No actionable WhatsApp message changes detected",
    )
