"""SQLite-backed approval queue for outgoing WhatsApp replies.

Every reply the agent drafts lands here as ``pending``; nothing reaches the
patient until a staff member approves it on the ``/staff`` page. Urgent
(escalated) drafts are flagged so they sort to the top.

Status lifecycle::

    pending --approve--> sending --delivered--> sent
                            \\--send failed--> pending
    pending --reject--> rejected

``sending`` is claimed atomically, so a double-clicked Approve can never
deliver the same draft twice.
"""

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

MAX_DRAFT_LENGTH = 4096  # WhatsApp text body limit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS drafts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sender TEXT NOT NULL,
    patient_message TEXT NOT NULL,
    draft_text TEXT NOT NULL,
    status TEXT NOT NULL,
    is_urgent INTEGER NOT NULL DEFAULT 0,
    timestamp TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts (status, is_urgent, timestamp);
"""

_COLUMNS = "id, sender, patient_message, draft_text, status, is_urgent, timestamp, updated_at"


class DraftStatus(str, Enum):
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Draft:
    id: int
    sender: str
    patient_message: str
    draft_text: str
    status: DraftStatus
    is_urgent: bool
    timestamp: str
    updated_at: str


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _validate_text(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        raise ValueError("draft text must not be empty")
    if len(stripped) > MAX_DRAFT_LENGTH:
        raise ValueError(f"draft text must be at most {MAX_DRAFT_LENGTH} characters")
    return stripped


class DraftQueue:
    """Thread-safe queue of reply drafts over one SQLite connection.

    ``db_path=":memory:"`` gives a private in-memory queue (used by tests).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(_SCHEMA)

    # -- Writes -------------------------------------------------------------

    def add(self, sender: str, patient_message: str, draft_text: str, is_urgent: bool) -> Draft:
        now = _utcnow()
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "INSERT INTO drafts (sender, patient_message, draft_text, status, is_urgent, timestamp, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (sender, patient_message, _validate_text(draft_text), DraftStatus.PENDING.value, int(is_urgent), now, now),
            )
            draft_id = cursor.lastrowid
        draft = self.get(draft_id)
        assert draft is not None
        return draft

    def update_text(self, draft_id: int, text: str) -> bool:
        """Replace a pending draft's text. Returns False if it is no longer pending."""
        return self._set(draft_id, "draft_text = ?", (_validate_text(text),), from_status=DraftStatus.PENDING)

    def claim_for_sending(self, draft_id: int, text: str) -> bool:
        """Atomically move ``pending -> sending`` with the final text. False if already taken."""
        return self._set(
            draft_id, "draft_text = ?, status = ?", (_validate_text(text), DraftStatus.SENDING.value), from_status=DraftStatus.PENDING
        )

    def mark_sent(self, draft_id: int) -> bool:
        return self._set(draft_id, "status = ?", (DraftStatus.SENT.value,), from_status=DraftStatus.SENDING)

    def release(self, draft_id: int) -> bool:
        """``sending -> pending`` after a failed delivery, so staff can retry."""
        return self._set(draft_id, "status = ?", (DraftStatus.PENDING.value,), from_status=DraftStatus.SENDING)

    def reject(self, draft_id: int) -> bool:
        return self._set(draft_id, "status = ?", (DraftStatus.REJECTED.value,), from_status=DraftStatus.PENDING)

    def _set(self, draft_id: int, assignments: str, values: tuple, from_status: DraftStatus) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                f"UPDATE drafts SET {assignments}, updated_at = ? WHERE id = ? AND status = ?",
                (*values, _utcnow(), draft_id, from_status.value),
            )
            return cursor.rowcount == 1

    # -- Reads --------------------------------------------------------------

    def get(self, draft_id: int) -> Optional[Draft]:
        with self._lock:
            row = self._conn.execute(f"SELECT {_COLUMNS} FROM drafts WHERE id = ?", (draft_id,)).fetchone()
        return _to_draft(row) if row is not None else None

    def list_pending(self) -> List[Draft]:
        """Pending drafts, urgent first, then oldest first."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM drafts WHERE status = ? ORDER BY is_urgent DESC, timestamp ASC, id ASC",
                (DraftStatus.PENDING.value,),
            ).fetchall()
        return [_to_draft(row) for row in rows]

    def list_recent(self, limit: int = 20) -> List[Draft]:
        """Recently handled (sent or rejected) drafts, newest first."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLUMNS} FROM drafts WHERE status IN (?, ?) ORDER BY updated_at DESC, id DESC LIMIT ?",
                (DraftStatus.SENT.value, DraftStatus.REJECTED.value, limit),
            ).fetchall()
        return [_to_draft(row) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _to_draft(row: sqlite3.Row) -> Draft:
    return Draft(
        id=row["id"],
        sender=row["sender"],
        patient_message=row["patient_message"],
        draft_text=row["draft_text"],
        status=DraftStatus(row["status"]),
        is_urgent=bool(row["is_urgent"]),
        timestamp=row["timestamp"],
        updated_at=row["updated_at"],
    )


__all__ = ["Draft", "DraftQueue", "DraftStatus", "MAX_DRAFT_LENGTH"]
