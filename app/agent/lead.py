"""Lead qualification domain model for the AI WhatsApp Support Agent.

This module owns three things:

- ``LeadProfile``: the validated, accumulating picture of who the customer
  is and what they want (universal + consumer + wholesale fields).
- ``LeadDelta``: a partial update to a ``LeadProfile`` — the shape a future
  extraction step will produce from a single user turn. Merging a delta is
  additive: it never erases an existing value with ``None``.
- ``evaluate_qualification``: the deterministic rule that computes a
  ``QualificationState`` from a ``LeadProfile``. Qualification is *derived*
  from validated data, never asserted by an LLM — there is no field on
  ``LeadDelta`` that can set it.

Design constraints (Milestone 2, Slice 4):
- No network. No LLM. No orchestrator. Pure data + rules.
- Explicit Pydantic models and enums, not loose dictionaries.
- Validation is deliberately modest: sensible lengths, legal enum values,
  sane numeric bounds, an email shape check, and non-blank names.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.knowledge import BrewMethod

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class LeadTrack(str, Enum):
    UNKNOWN = "unknown"
    CONSUMER = "consumer"
    WHOLESALE = "wholesale"


class QualificationState(str, Enum):
    UNKNOWN = "unknown"
    BROWSING = "browsing"
    COLLECTING = "collecting"
    QUALIFIED = "qualified"
    HANDOFF_READY = "handoff_ready"
    DECLINED = "declined"
    ESCALATED = "escalated"


class BusinessType(str, Enum):
    """Wholesale customer types, mirroring ``wholesale_info.customer_types``
    in ``data/business.json`` plus an explicit catch-all."""

    CAFE = "cafe"
    OFFICE = "office"
    RESTAURANT = "restaurant"
    RESELLER = "reseller"
    EVENTS = "events"
    OTHER = "other"


class Timeline(str, Enum):
    IMMEDIATE = "immediate"
    WITHIN_1_MONTH = "within_1_month"
    WITHIN_3_MONTHS = "within_3_months"
    EXPLORING = "exploring"


class BudgetBand(str, Enum):
    """Consumer budget per order, in INR."""

    UNDER_500 = "under_500"
    FROM_500_TO_1000 = "500_1000"
    FROM_1000_TO_2000 = "1000_2000"
    ABOVE_2000 = "above_2000"


class LeadSource(str, Enum):
    WHATSAPP = "whatsapp"
    MANUAL = "manual"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sensible upper bound for a monthly wholesale order (kg). The largest real
# wholesale accounts are far below this; anything above is almost certainly
# an extraction error (e.g. grams parsed as kg) rather than a real requirement.
MAX_MONTHLY_VOLUME_KG = 10_000.0

# Field groups. These drive both qualification and provenance validation, so
# they are defined once here rather than repeated in each rule.
UNIVERSAL_FIELDS: Tuple[str, ...] = (
    "contact_name",
    "email",
    "city",
    "intent_summary",
)
CONSUMER_FIELDS: Tuple[str, ...] = (
    "brew_method",
    "taste_preference",
    "budget_band",
    "subscription_interest",
)
WHOLESALE_FIELDS: Tuple[str, ...] = (
    "business_name",
    "business_type",
    "monthly_volume_kg",
    "timeline",
    "current_supplier",
)
# Fields a delta may set and provenance may reference. ``whatsapp_number`` is
# excluded on purpose: it is webhook metadata, never extracted from text.
LEAD_DATA_FIELDS: Tuple[str, ...] = ("track",) + UNIVERSAL_FIELDS + CONSUMER_FIELDS + WHOLESALE_FIELDS

WHOLESALE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "contact_name",
    "business_name",
    "business_type",
    "monthly_volume_kg",
    "city",
    "timeline",
)
CONSUMER_REQUIRED_FIELDS: Tuple[str, ...] = (
    "contact_name",
    "brew_method",
    "taste_preference",
)

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

# Deliberately simple: one "@", something either side, a dot in the domain.
# This is a sanity check on extracted text, not an RFC 5322 validator.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_WHATSAPP_NUMBER_RE = re.compile(r"^\d{6,20}$")


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


def _validate_email(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    if not _EMAIL_RE.match(stripped):
        raise ValueError("email is not a valid address")
    return stripped.lower()


# ---------------------------------------------------------------------------
# LeadProfile
# ---------------------------------------------------------------------------


class LeadProfile(BaseModel):
    """Everything known about the lead so far.

    All data fields default to ``None`` so a fresh profile is valid and
    "nothing known" is represented explicitly. ``field_provenance`` records
    the conversation turn on which each field was last set, so a later
    slice can explain *why* the agent believes something.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    track: LeadTrack = LeadTrack.UNKNOWN

    # Universal
    contact_name: Optional[str] = Field(None, max_length=80)
    whatsapp_number: Optional[str] = Field(None, max_length=20)
    email: Optional[str] = Field(None, max_length=120)
    city: Optional[str] = Field(None, max_length=80)
    intent_summary: Optional[str] = Field(None, max_length=300)

    # Consumer
    brew_method: Optional[BrewMethod] = None
    taste_preference: Optional[str] = Field(None, max_length=120)
    budget_band: Optional[BudgetBand] = None
    subscription_interest: Optional[bool] = None

    # Wholesale
    business_name: Optional[str] = Field(None, max_length=120)
    business_type: Optional[BusinessType] = None
    monthly_volume_kg: Optional[float] = Field(None, ge=0, le=MAX_MONTHLY_VOLUME_KG)
    timeline: Optional[Timeline] = None
    current_supplier: Optional[str] = Field(None, max_length=120)

    # Metadata
    source: LeadSource = LeadSource.WHATSAPP
    field_provenance: Dict[str, int] = Field(default_factory=dict)

    @field_validator("contact_name", "business_name")
    @classmethod
    def _names_not_blank(cls, value: Optional[str], info) -> Optional[str]:
        return _reject_blank(info.field_name, value)

    @field_validator("city", "intent_summary", "taste_preference", "current_supplier")
    @classmethod
    def _free_text_strip(cls, value: Optional[str]) -> Optional[str]:
        return _strip_or_none(value)

    @field_validator("email")
    @classmethod
    def _email_shape(cls, value: Optional[str]) -> Optional[str]:
        return _validate_email(value)

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

    def effective_track(self) -> LeadTrack:
        """The track to qualify against.

        An explicitly set ``track`` wins. Otherwise infer from the data:
        any wholesale field present means wholesale; any consumer field
        present means consumer; nothing means unknown.
        """
        if self.track != LeadTrack.UNKNOWN:
            return self.track
        if any(getattr(self, name) is not None for name in WHOLESALE_FIELDS):
            return LeadTrack.WHOLESALE
        if any(getattr(self, name) is not None for name in CONSUMER_FIELDS):
            return LeadTrack.CONSUMER
        return LeadTrack.UNKNOWN

    def has_any_lead_data(self) -> bool:
        """True once anything beyond webhook metadata has been collected."""
        return any(getattr(self, name) is not None for name in LEAD_DATA_FIELDS if name != "track") or (
            self.track != LeadTrack.UNKNOWN
        )

    def missing_required_fields(self) -> List[str]:
        """Required fields (for the effective track) that are still ``None``.

        For an unknown track this is the consumer requirement set, because
        that is the minimum any lead needs and it keeps the answer
        deterministic rather than empty.
        """
        track = self.effective_track()
        required = WHOLESALE_REQUIRED_FIELDS if track == LeadTrack.WHOLESALE else CONSUMER_REQUIRED_FIELDS
        return [name for name in required if getattr(self, name) is None]

    def is_complete(self) -> bool:
        """Whether the profile satisfies the requirement set for its track.

        A profile with no determinable track is never complete.
        """
        if self.effective_track() == LeadTrack.UNKNOWN:
            return False
        return not self.missing_required_fields()


# ---------------------------------------------------------------------------
# LeadDelta
# ---------------------------------------------------------------------------


class LeadDelta(BaseModel):
    """A partial update to a ``LeadProfile`` from one conversation turn.

    Every field is optional; ``None`` means "no new information", never
    "clear this". ``extra="forbid"`` guarantees an extractor (or a model
    output) cannot smuggle in fields that are not lead data — in
    particular there is no ``qualification`` field here, by design.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    track: Optional[LeadTrack] = None

    contact_name: Optional[str] = Field(None, max_length=80)
    email: Optional[str] = Field(None, max_length=120)
    city: Optional[str] = Field(None, max_length=80)
    intent_summary: Optional[str] = Field(None, max_length=300)

    brew_method: Optional[BrewMethod] = None
    taste_preference: Optional[str] = Field(None, max_length=120)
    budget_band: Optional[BudgetBand] = None
    subscription_interest: Optional[bool] = None

    business_name: Optional[str] = Field(None, max_length=120)
    business_type: Optional[BusinessType] = None
    monthly_volume_kg: Optional[float] = Field(None, ge=0, le=MAX_MONTHLY_VOLUME_KG)
    timeline: Optional[Timeline] = None
    current_supplier: Optional[str] = Field(None, max_length=120)

    @field_validator("contact_name", "business_name")
    @classmethod
    def _names_not_blank(cls, value: Optional[str], info) -> Optional[str]:
        return _reject_blank(info.field_name, value)

    @field_validator("city", "intent_summary", "taste_preference", "current_supplier")
    @classmethod
    def _free_text_strip(cls, value: Optional[str]) -> Optional[str]:
        return _strip_or_none(value)

    @field_validator("email")
    @classmethod
    def _email_shape(cls, value: Optional[str]) -> Optional[str]:
        return _validate_email(value)

    def provided_fields(self) -> Dict[str, object]:
        """Fields carrying a value (i.e. not ``None``), in declaration order."""
        return {name: getattr(self, name) for name in LEAD_DATA_FIELDS if getattr(self, name) is not None}

    def is_empty(self) -> bool:
        return not self.provided_fields()


def merge_lead_delta(profile: LeadProfile, delta: LeadDelta, turn: int) -> LeadProfile:
    """Return a new ``LeadProfile`` with ``delta`` applied at ``turn``.

    Semantics:
    - Only fields present (non-``None``) in the delta are touched, so an
      existing value is never replaced by ``None``.
    - A ``track`` of ``unknown`` in the delta is treated as "no information"
      and does not overwrite a known track.
    - Each applied field gets ``field_provenance[field] = turn``.
    - The result is re-validated as a whole; the input profile is untouched.

    Raises:
        ValueError: if ``turn`` is negative or the merged profile is invalid.
    """
    if turn < 0:
        raise ValueError("turn must be a non-negative turn number")

    updates = delta.provided_fields()
    if updates.get("track") == LeadTrack.UNKNOWN:
        updates.pop("track")

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

    This is the single place qualification is decided. It is a pure
    function: same inputs, same output, and nothing here reads model
    output or free text.

    Rules, in order:
    1. Sticky states (``handoff_ready``, ``declined``, ``escalated``) are
       returned unchanged — leaving them is an explicit action, not a
       data question.
    2. A complete profile for its track is ``qualified``.
    3. Any collected lead data (but not yet complete) is ``collecting``.
    4. No lead data but at least one turn taken is ``browsing``.
    5. Otherwise ``unknown``.

    Because ``merge_lead_delta`` never removes data, the data-driven states
    only move forward in practice; the rule set still recomputes from
    scratch so it stays honest if a profile is edited directly.
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
    "BudgetBand",
    "BusinessType",
    "CONSUMER_FIELDS",
    "CONSUMER_REQUIRED_FIELDS",
    "LEAD_DATA_FIELDS",
    "LeadDelta",
    "LeadProfile",
    "LeadSource",
    "LeadTrack",
    "MAX_MONTHLY_VOLUME_KG",
    "QualificationState",
    "STICKY_QUALIFICATION_STATES",
    "Timeline",
    "UNIVERSAL_FIELDS",
    "WHOLESALE_FIELDS",
    "WHOLESALE_REQUIRED_FIELDS",
    "evaluate_qualification",
    "merge_lead_delta",
]
