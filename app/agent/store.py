"""In-memory ``ConversationState`` store with LRU eviction.

``ConversationStore`` is deliberately tiny: a bounded ``OrderedDict`` keyed
by sender ID. It holds validated ``ConversationState`` snapshots and hands
out copies, so a caller can never mutate stored state without going back
through ``save``/``update``. That keeps "what is persisted" equal to "what
passed validation".

Design constraints (Milestone 2, Slice 4):
- In-memory only. No database, no Redis, no external service, no network.
- Bounded to ``STORE_CAPACITY`` (500) conversations; the least recently
  used sender is evicted when a new one would exceed the cap.
- The Milestone 1 ``InMemoryConversationMemory`` (chat buffer +
  idempotency ledger) is untouched. This store sits beside it; wiring the
  two together is orchestrator work for a later slice.
"""

import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Optional

from app.agent.state import STORE_CAPACITY, ConversationState
from app.config import mask_phone_number

logger = logging.getLogger(__name__)


class ConversationStore:
    """Bounded, LRU-evicting, in-memory store of ``ConversationState``."""

    def __init__(self, capacity: int = STORE_CAPACITY):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._states: "OrderedDict[str, ConversationState]" = OrderedDict()

    # -- Read ---------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, sender_id: str) -> bool:
        return sender_id in self._states

    def sender_ids(self) -> List[str]:
        """Sender IDs from least to most recently used."""
        return list(self._states.keys())

    def get(self, sender_id: str) -> Optional[ConversationState]:
        """Return a copy of the sender's state, or ``None`` if unknown.

        A hit counts as a use for LRU purposes.
        """
        state = self._states.get(sender_id)
        if state is None:
            return None
        self._states.move_to_end(sender_id)
        return state.model_copy(deep=True)

    def load(self, sender_id: str) -> Optional[ConversationState]:
        """Alias of :meth:`get`, paired with :meth:`save`."""
        return self.get(sender_id)

    def get_or_create(self, sender_id: str) -> ConversationState:
        """Return the sender's state, creating and saving a safe default
        (``ConversationState.new``) if none exists yet."""
        existing = self.get(sender_id)
        if existing is not None:
            return existing
        state = ConversationState.new(sender_id)
        self.save(state)
        return state

    # -- Write --------------------------------------------------------------

    def save(self, state: ConversationState) -> None:
        """Persist a validated snapshot of ``state`` under its sender ID.

        The stored object is a re-validated copy: a caller holding the
        original can keep mutating it without affecting the store until the
        next ``save``.
        """
        snapshot = ConversationState.model_validate(state.model_dump())
        sender_id = snapshot.sender_id
        if sender_id in self._states:
            self._states.move_to_end(sender_id)
        self._states[sender_id] = snapshot
        self._evict_if_needed()

    def update(
        self,
        sender_id: str,
        mutator: Callable[[ConversationState], None],
    ) -> ConversationState:
        """Load (or create), apply ``mutator`` in place, save, return a copy.

        ``mutator`` receives a working copy; nothing is persisted if it
        raises, so a failed validation leaves the store unchanged.
        """
        state = self.get_or_create(sender_id)
        mutator(state)
        self.save(state)
        return state.model_copy(deep=True)

    def delete(self, sender_id: str) -> bool:
        """Remove a sender's state. Returns whether anything was removed."""
        return self._states.pop(sender_id, None) is not None

    def clear(self) -> None:
        self._states.clear()

    # -- Serialization ------------------------------------------------------

    def snapshot(self) -> Dict[str, dict]:
        """JSON-compatible dump of every stored state, keyed by sender ID,
        in least-to-most-recently-used order."""
        return {sender_id: state.model_dump(mode="json") for sender_id, state in self._states.items()}

    @classmethod
    def from_snapshot(cls, data: Dict[str, dict], capacity: int = STORE_CAPACITY) -> "ConversationStore":
        """Rebuild a store from :meth:`snapshot` output.

        Entries are re-validated in order, so LRU ordering survives the
        round-trip and a corrupt entry fails loudly instead of loading
        silently.
        """
        store = cls(capacity=capacity)
        for sender_id, payload in data.items():
            state = ConversationState.model_validate(payload)
            if state.sender_id != sender_id:
                raise ValueError(
                    f"snapshot key {mask_phone_number(sender_id)} does not match "
                    f"state sender_id {mask_phone_number(state.sender_id)}"
                )
            store.save(state)
        return store

    # -- Internal -----------------------------------------------------------

    def _evict_if_needed(self) -> None:
        while len(self._states) > self.capacity:
            evicted_id, _ = self._states.popitem(last=False)
            logger.info(
                "ConversationStore at capacity (%d); evicted least recently used sender %s",
                self.capacity,
                mask_phone_number(evicted_id),
            )


__all__ = ["ConversationStore"]
