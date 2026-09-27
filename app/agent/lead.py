"""Patient lead (booking request) domain model for the AI WhatsApp Support Agent.

This module owns three things:

- ``LeadProfile``: the validated, accumulating picture of the patient and
  the appointment they want: name, callback phone, concern, and preferred
  day/time. It fills in incrementally across turns.
- ``LeadDelta``: a partial update to a ``LeadProfile``, produced from a
  single patient message. Merging a delta is additive: it never erases an
  existing value with ``None``.
- ``evaluate_qualification``: the deterministic rule that computes a
  ``QualificationState`` from a ``LeadProfile``. Qualification is *derived*
  from validated data, never asserted by an LLM — there is no field on
  ``LeadDelta`` that can set it.

The patient's WhatsApp number is webhook metadata. It is the default
callback phone, so the assistant never has to ask for a phone number; a
number the patient types in explicitly overrides it.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class QualificationState(str, Enum):
    UNKNOWN = "unknown"
    BROWSING = "browsing"
    COLLECTING = "collecting"
    QUALIFIED = "qualified"
    HANDOFF_READY = "handoff_ready"
    DECLINED = "declined"
    ESCALATED = "escalated"


class LeadSource(str, Enum):
    WHATSAPP = "whatsapp"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fields a delta may set and provenance may reference. ``whatsapp_number`` is
# excluded on purpose: it is webhook metadata, never extracted from text.
LEAD_DATA_FIELDS: Tuple[str, ...] = ("patient_name", "phone", "concern", "preferred_day_time")
REQUIRED_FIELDS: Tuple[str, ...] = LEAD_DATA_FIELDS

# The order in which missing booking details are asked for, one per reply.
QUESTION_ORDER: Tuple[str, ...] = ("concern", "patient_name", "preferred_day_time", "phone")

# The single question PromptBuilder may authorize for a missing field. The
# model phrases it in the patient's language; the intent must not change.
BOOKING_QUESTIONS: Dict[str, str] = {
    "concern": "What would you like to see the dentist about (for example a check-up, cleaning, or a specific problem)?",
    "patient_name": "May I have the patient's name for the booking?",
    "preferred_day_time": "Which day and time would suit you for the visit?",
    "phone": "Which phone number should the clinic call to confirm the appointment?",
}

# States that are only entered by an explicit decision (user consent,
# a decline, or an escalation) and are therefore never overwritten by a
# data-driven re-evaluation of the profile.
STICKY_QUALIFICATION_STATES = frozenset(
    {
        QualificationState.HANDOFF_READY,
        QualificationState.DECLINED,
        QualificationState.ESCALATED,
    }
)

_WHATSAPP_NUMBER_RE = re.compile(r"^\d{6,20}$")
_PHONE_SEPARATORS_RE = re.compile(r"[\s().-]")
_PHONE_RE = re.compile(r"^\+?\d{10,15}$")


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------


def _strip_or_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _reject_blank(field_name: str, value: Optional[str]) -> Optional[str]:
    """Trim; reject a value that was present but whitespace-only."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return stripped


def _normalize_phone(value: Optional[str]) -> Optional[str]:
    """Digits only (a leading ``+`` is dropped); 10-15 digits, else invalid."""
    if value is None:
        return None
    compact = _PHONE_SEPARATORS_RE.sub("", value.strip())
    if not _PHONE_RE.match(compact):
        raise ValueError("phone must contain 10-15 digits")
    return compact.lstrip("+")


# ---------------------------------------------------------------------------
# LeadProfile
# ---------------------------------------------------------------------------


class LeadProfile(BaseModel):
    """Everything known about the patient's booking request so far.

    All data fields default to ``None`` so a fresh profile is valid and
    "nothing known" is represented explicitly. ``field_provenance`` records
    the conversation turn on which each field was last set.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patient_name: Optional[str] = Field(None, max_length=80)
    phone: Optional[str] = Field(None, max_length=20)
    concern: Optional[str] = Field(None, max_length=200)
    preferred_day_time: Optional[str] = Field(None, max_length=80)

    # Metadata
    whatsapp_number: Optional[str] = Field(None, max_length=20)
    source: LeadSource = LeadSource.WHATSAPP
    field_provenance: Dict[str, int] = Field(default_factory=dict)

    @field_validator("patient_name")
    @classmethod
    def _name_not_blank(cls, value: Optional[str], info) -> Optional[str]:
        return _reject_blank(info.field_name, value)

    @field_validator("concern", "preferred_day_time")
    @classmethod
    def _free_text_strip(cls, value: Optional[str]) -> Optional[str]:
        return _strip_or_none(value)

    @field_validator("phone")
    @classmethod
    def _phone_shape(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_phone(value)

    @field_validator("whatsapp_number")
    @classmethod
    def _whatsapp_number_shape(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip()
        if not _WHATSAPP_NUMBER_RE.match(stripped):
            raise ValueError("whatsapp_number must be 6-20 digits")
        return stripped

    @model_validator(mode="after")
    def _provenance_is_consistent(self) -> "LeadProfile":
        for name, turn in self.field_provenance.items():
            if name not in LEAD_DATA_FIELDS:
                raise ValueError(f"field_provenance references unknown field '{name}'")
            if not isinstance(turn, int) or isinstance(turn, bool) or turn < 0:
                raise ValueError(
                    f"field_provenance['{name}'] must be a non-negative turn number, got {turn!r}"
                )
            if getattr(self, name) is None:
                raise ValueError(f"field_provenance['{name}'] set but the field is None")
        return self

    # -- Derived views ------------------------------------------------------

    def callback_phone(self) -> Optional[str]:
        """The number the clinic should call: the patient's stated phone, else WhatsApp."""
        return self.phone or self.whatsapp_number

    def has_any_lead_data(self) -> bool:
        """True once the patient has told us anything (webhook metadata excluded)."""
        return any(getattr(self, name) is not None for name in LEAD_DATA_FIELDS)

    def missing_required_fields(self) -> List[str]:
        """Required booking fields still unknown, in ``REQUIRED_FIELDS`` order."""
        missing: List[str] = []
        for name in REQUIRED_FIELDS:
            value = self.callback_phone() if name == "phone" else getattr(self, name)
            if value is None:
                missing.append(name)
        return missing

    def is_complete(self) -> bool:
        return not self.missing_required_fields()

    def next_missing_field(self) -> Optional[str]:
        """The one field to ask about next, or ``None`` when the booking request is complete."""
        missing = set(self.missing_required_fields())
        return next((name for name in QUESTION_ORDER if name in missing), None)


# ---------------------------------------------------------------------------
# LeadDelta
# ---------------------------------------------------------------------------


class LeadDelta(BaseModel):
    """A partial update to a ``LeadProfile`` from one patient message.

    Every field is optional; ``None`` means "no new information", never
    "clear this". ``extra="forbid"`` guarantees an extractor (or a model
    output) cannot smuggle in fields that are not lead data — in
    particular there is no ``qualification`` field here, by design.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    patient_name: Optional[str] = Field(None, max_length=80)
    phone: Optional[str] = Field(None, max_length=20)
    concern: Optional[str] = Field(None, max_length=200)
    preferred_day_time: Optional[str] = Field(None, max_length=80)

    @field_validator("patient_name")
    @classmethod
    def _name_not_blank(cls, value: Optional[str], info) -> Optional[str]:
        return _reject_blank(info.field_name, value)

    @field_validator("concern", "preferred_day_time")
    @classmethod
    def _free_text_strip(cls, value: Optional[str]) -> Optional[str]:
        return _strip_or_none(value)

    @field_validator("phone")
    @classmethod
    def _phone_shape(cls, value: Optional[str]) -> Optional[str]:
        return _normalize_phone(value)

    def provided_fields(self) -> Dict[str, object]:
        """Fields carrying a value (i.e. not ``None``), in declaration order."""
        return {name: getattr(self, name) for name in LEAD_DATA_FIELDS if getattr(self, name) is not None}

    def is_empty(self) -> bool:
        return not self.provided_fields()


def merge_lead_delta(profile: LeadProfile, delta: LeadDelta, turn: int) -> LeadProfile:
    """Return a new ``LeadProfile`` with ``delta`` applied at ``turn``.

    Only fields present (non-``None``) in the delta are touched, so an
    existing value is never replaced by ``None``. Each applied field gets
    ``field_provenance[field] = turn``. The input profile is untouched.

    Raises:
        ValueError: if ``turn`` is negative or the merged profile is invalid.
    """
    if turn < 0:
        raise ValueError("turn must be a non-negative turn number")

    updates = delta.provided_fields()
    if not updates:
        return profile.model_copy(deep=True)

    data = profile.model_dump()
    provenance = dict(data["field_provenance"])
    for name, value in updates.items():
        data[name] = value
        provenance[name] = turn
    data["field_provenance"] = provenance
    return LeadProfile.model_validate(data)


# ---------------------------------------------------------------------------
# Qualification evaluation
# ---------------------------------------------------------------------------


def evaluate_qualification(
    profile: LeadProfile,
    current: QualificationState,
    turn_count: int,
) -> QualificationState:
    """Compute the next ``QualificationState`` from validated profile data.

    Rules, in order:
    1. Sticky states (``handoff_ready``, ``declined``, ``escalated``) are
       returned unchanged — leaving them is an explicit action.
    2. A complete booking request is ``qualified``.
    3. Any collected lead data (but not yet complete) is ``collecting``.
    4. No lead data but at least one turn taken is ``browsing``.
    5. Otherwise ``unknown``.
    """
    if current in STICKY_QUALIFICATION_STATES:
        return current
    if profile.is_complete():
        return QualificationState.QUALIFIED
    if profile.has_any_lead_data():
        return QualificationState.COLLECTING
    if turn_count > 0:
        return QualificationState.BROWSING
    return QualificationState.UNKNOWN


__all__ = [
    "BOOKING_QUESTIONS",
    "LEAD_DATA_FIELDS",
    "LeadDelta",
    "LeadProfile",
    "LeadSource",
    "QUESTION_ORDER",
    "QualificationState",
    "REQUIRED_FIELDS",
    "STICKY_QUALIFICATION_STATES",
    "evaluate_qualification",
    "merge_lead_delta",
]
