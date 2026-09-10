import logging
from collections import deque
from typing import Dict, List
from app.config import mask_phone_number
from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)


class InMemoryConversationMemory:
    """In-memory conversation history buffer keyed by sender ID/phone number."""

    def __init__(self, max_messages: int = 10, max_seen_message_ids: int = 500):
        self.max_messages = max_messages
        self._conversations: Dict[str, deque] = {}
        # Idempotency ledger: bounded FIFO of recently processed Meta message IDs (wamid),
        # so a Meta webhook retry for the same event does not trigger a duplicate AI reply.
        self._seen_message_ids: deque = deque(maxlen=max_seen_message_ids)
        self._seen_message_id_set: set = set()

    def has_processed(self, message_id: str) -> bool:
        """Return True if this Meta message ID has already been handled."""
        return message_id in self._seen_message_id_set

    def mark_processed(self, message_id: str) -> None:
        """Record a Meta message ID as handled, evicting the oldest entry if full."""
        if message_id in self._seen_message_id_set:
            return
        if len(self._seen_message_ids) == self._seen_message_ids.maxlen:
            oldest = self._seen_message_ids.popleft()
            self._seen_message_id_set.discard(oldest)
        self._seen_message_ids.append(message_id)
        self._seen_message_id_set.add(message_id)

    def get_messages(self, sender_id: str) -> List[ChatMessage]:
        """Retrieve recent messages for a given sender.
        
        Args:
            sender_id: The phone number or sender identifier.
            
        Returns:
            List of ChatMessage objects in chronological order.
        """
        if sender_id not in self._conversations:
            return []
        return list(self._conversations[sender_id])

    def add_user_message(self, sender_id: str, content: str) -> None:
        """Record an incoming user message into the sender's history."""
        self._append(sender_id, ChatMessage(role="user", content=content))

    def add_assistant_message(self, sender_id: str, content: str) -> None:
        """Record an outgoing assistant reply into the sender's history."""
        self._append(sender_id, ChatMessage(role="assistant", content=content))

    def _append(self, sender_id: str, message: ChatMessage) -> None:
        if sender_id not in self._conversations:
            self._conversations[sender_id] = deque(maxlen=self.max_messages)
        self._conversations[sender_id].append(message)
        logger.debug(
            "Appended message for %s (current length: %d)",
            mask_phone_number(sender_id),
            len(self._conversations[sender_id]),
        )

    def clear(self, sender_id: str) -> None:
        """Clear the history for a specific sender."""
        if sender_id in self._conversations:
            self._conversations[sender_id].clear()
            del self._conversations[sender_id]

    def clear_all(self) -> None:
        """Clear all conversation histories."""
        self._conversations.clear()
